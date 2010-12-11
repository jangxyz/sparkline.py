
"""spark.py
A python module for generating sparklines.
Requires the Python Imaging Library 
"""

__author__ = "Joe Gregorio (joe@bitworking.org), Matthew Perry (perrygeo@gmail.com"
__copyright__ = "Copyright 2005, Joe Gregorio"
__contributors__ = ['Alan Powell, Matthew Perry']
__version__ = "0.1"
__license__ = "MIT"
__history__ = """

20070510 abstracted functions away from cgi-specific arg objects (MTP)

"""

import Image, ImageDraw
import types

SequenceTypes = (types.ListType, types.TupleType)

def draw(data, **options):
    chart = options.get('type')
    chart_class = globals().get(chart, LineChart)
    return chart_class

class Image:
    def draw(self, data, **options):
        pass



def composite(s1, s2):
    pass

def merge_dict(d1, d2):
    d = {}
    for k in d1: d[k] = d1[k]
    for k in d2: d[k] = d2[k]
    return d

def extract_dict(d, keys):
    new_d = {}
    for k in keys: new_d[k] = d[k]
    return new_d


class Sparkline(object):
    common_options = { 
        'type': None,
        'width': None,
        'height': None,
        'line_color': "#888888",
        'fill_color': False,
        'chart_range_min': None,
        'chart_range_max': None,
        'composite': None, #?
    }


    def __init__(self, **options):
        for opt, default in self.__class__.common_options.items():
            setattr(self, opt, options.get(opt, default))

    def set_options(self, **options):
        for opt, default in self.__class__.options.iteritems():
            setattr(self, opt, options.get(opt, default))

    def min(self, data):
        return min(y for y in data if y is not None)

    def resolve_options(self, data, ignore=[]):
        options = []
        # data may be [3,5,2] or [(1,3), (2,5), (3,2)]
        if isinstance(data[0], SequenceTypes):
            data = [y for (x,y) in data]

        if self.chart_range_min is None and 'chart_range_min' not in ignore: 
            options['chart_range_min'] = self.min(data)
        if self.chart_range_max is None and 'chart_range_max' not in ignore: 
            options['chart_range_max'] = max(data)

        if self.height is None and 'height' not in ignore:
            options['height'] = self.chart_range_max - self.chart_range_min

        return extract_dict(self.__dict__, self.common_options.keys())

    def draw(self):
        pass

    def render(self, output=None):
        if output:
            self.image.save(output, "PNG")
            return output
        else:
            return self.image



class LineChart(Sparkline):
    options = {
        'default_pixels_per_value': None,
        'spot_color': '#f80',
        'min_spot_color': '#f80',
        'max_spot_color': '#f80',
        'spot_radius': 1.5,
        'line_width': 1,
        'normal_range_min': None, 
        'normal_range_max': None, 
        'normal_range_color': '#ccc', 
        'xvalues': None,
        'chart_range_clip':   False, #?
        'chart_range_clip_x': False, #?
        'chart_range_min_x': None, #?
        'chart_range_max_x': None, #?
    }

    def __init__(self, **options):
        super(LineChart, self).__init__(type='line', **options)
        self.set_options(**options)
        self.image = None

    def set_options(self, **options):
        super(LineChart, self).set_options(**options)
        self.spot_radius = int(round(self.spot_radius))

    def resolve_options(self, data):
        options = {}
        if isinstance(data[0], SequenceTypes):
            data = [y for (x,y) in data]
            options['xvalues'] = [x for (x,y) in data]

        if self.default_pixels_per_value is None:
            if self.width:  options['default_pixels_per_value'] = self.width / len(data)
            else:           options['default_pixels_per_value'] = 3

        if self.max_spot_color is True: options['max_spot_color'] = self.spot_color
        if self.min_spot_color is True: options['min_spot_color'] = self.spot_color

        if self.normal_range_min or self.normal_range_max:
            if self.normal_range_min is None: options['normal_range_min'] = self.min(data)
            if self.normal_range_max is None: options['normal_range_max'] = max(data)

        # set common options
        common_options = super(LineChart, self).resolve_options(data)

        return merge_dict(common_options, options)


    def compute_y(self, y, options):
        height = options['height']
        chart_range_min = options['chart_range_min']
        chart_range_max = options['chart_range_max']
        scale = float(chart_range_max - chart_range_min + 1)/(height - 4)
        return height - 3  - (y - chart_range_min) / scale

    def draw(self, data):
        options = self.resolve_options(data)

        # compute coords, removing any None data
        xcoords = options['xvalues'] or range(len(data))
        xcoords = [1 + x * options['default_pixels_per_value'] for x in xcoords]
        ycoords = [self.compute_y(y, options) if y is not None else None for y in data]
        coords = zip(xcoords, ycoords)

        canvas_size = ((len(data)-1) * options['default_pixels_per_value'] + 4, options['height'])
        self.image = Image.new("RGB", canvas_size, 'white')
        draw = ImageDraw.Draw(self.image)
        if draw:
            # fill normal range
            if options['normal_range_min']:
                y1 = compute_y(options['normal_range_min'], options)
                y2 = compute_y(options['normal_range_max'], options)
                normal_range = [coords[0][0],y1, coords[-1][0],y2]
                draw.rectangle(normal_range, fill=options['normal_range_color'])

            def draw_except_void(draw, xy, draw_func):
                ''' draw except the None area '''
                start = 0
                for end in (i for i,c in enumerate(xy) if c[1] is None):
                    draw_func(draw, xy[start:end])
                    start = end+1
                if xy[start:]:
                    draw_func(draw, xy[start:])

            # fill color
            if options['fill_color']:
                draw_poly = lambda draw, xy: draw.polygon([(xy[0][0],options['chart_range_max'])] + xy + [(xy[-1][0],options['chart_range_max'])], fill=options['fill_color'])
                draw_except_void(draw, coords, draw_poly)

            # draw line
            draw_line = lambda draw, xy: draw.line(xy, fill=options['line_color'], width=options['line_width'])
            draw_except_void(draw, coords, draw_line)

            # draw spots
            draw_spot = lambda draw, pt, color: \
                draw.ellipse([pt[0]-options['spot_radius'], pt[1]-options['spot_radius'], pt[0]+options['spot_radius'], pt[1]+options['spot_radius']], fill=color)
            if options['min_spot_color']:
                draw_spot(draw, coords[data.index(self.min(data))], options['min_spot_color'])
            if options['max_spot_color']:
                draw_spot(draw, coords[data.index(max(data))], options['max_spot_color'])
            if options['spot_color']:
                last_xy = max((xy for xy in coords if xy[1] is not None), key=lambda (x,y): x)
                draw_spot(draw, last_xy, options['spot_color'])
                
        del draw

        return self



class BarChart(Sparkline):
    options = {
        'bar_color': '#00f',     # Colour used for postive values
        'neg_bar_color': '#f44', # Colour used for negative values
        'zero_color': None,    # Colour used for values equal to zero
        'null_color': None,    #?
        'bar_width': 4,        # Width of each bar, in pixels
        'bar_spacing': 1,      # Space between each bar, in pixels
        'zero_axis': None,     # Centers the y-axis at zero if true (default is to automatically do the right thing)
        'color_map': None,     # Map override colors to certain values 
    }

    def __init__(self, **options):
        super(BarChart, self).__init__(type='bar', **options)
        self.set_options(**options)
        self.image = None

    def set_options(self, **options):
        super(BarChart, self).set_options(**options)

    def resolve_options(self, data):
        # set common options
        super(BarChart, self).resolve_options(data, ignore=['chart_range_min'])

        # zero axis
        has_negative_data = any(True for y in data if y is not None and y < 0)
        if self.zero_axis is None:
            self.zero_axis = has_negative_data is False

        # chart_range_min
        if self.chart_range_min is None: 
            if self.zero_axis is True:
                min_data = min(abs(y) for y in data if y is not None)
            else:
                min_data = min(y for y in data if y is not None)
            self.chart_range_min = min_data

        option_keys = self.common_options.keys() + self.options.keys()
        return extract_dict(self.__dict__, option_keys)


    def compute_y(self, y):
        scale = float(self.chart_range_max - self.chart_range_min + 1)/(self.height - 4)

        if self.zero_axis is True:
            return self.height - 3  - (abs(y) - self.chart_range_min) / scale
        else:
            return (self.height - 3)/2  - y / scale


    def draw(self, data):
        self.resolve_options(data)

        canvas_width = len(data) * (self.bar_width + self.bar_spacing + 1) - self.bar_spacing
        self.image = Image.new("RGB", (canvas_width, self.height), 'white')

        draw = ImageDraw.Draw(self.image)
        if draw:
            for i,y in enumerate(data):
                if y is None:
                    continue

                x  = i * (self.bar_width + self.bar_spacing + 1)
                _y = int(self.compute_y(y))

                color = self.select_color(i,y)
                end = (self.height) if self.zero_axis else (self.height - 3) / 2
                draw.rectangle([x,_y, x+self.bar_width,end], fill=color)
                
        del draw

        return self

    def select_color(self, i, y):
        # 
        if y >= 0:  color = self.bar_color
        else:       color = self.neg_bar_color

        # zero color
        if y is 0 and self.zero_color is not None:
            color = self.zero_color

        ## null color
        #if y is None and self.null_color is not None:
        #    color = self.null_color

        # color map
        if self.color_map is not None:
            # {-1: 'red', 0: 'green', 1: 'blue'}
            if isinstance(self.color_map, types.DictType):
                color = self.color_map.get(y, color)
            # ['red', 'green', 'blue']
            elif isinstance(self.color_map, SequenceTypes):
                if i < len(self.color_map):
                    color = self.color_map[i]

        return color

    

class Discrete(Sparkline):
    def run(self, data, output=None, 
        dmin=None, dmax=None, upper=None, width=2, height=14, 
        below_color='gray', above_color='red', longlines=False):
        gap = 4
        if longlines:
            gap = 0
 
        if dmin is None:  dmin = min(data)
        if dmax is None:  dmax = max(data)
        # defaults to the mean
        if upper is None: upper = sum(data) / len(data)

        if dmax < dmin:
            dmax = dmin

        image = Image.new("RGB", (len(data)*width-1, height), 'white') 
        zero = image.size[1] - 1
        if dmin < 0 and dmax > 0:
            zero = image.size[1] - (0 - dmin) / (float(dmax - dmin + 1) / (height - gap))

        draw = ImageDraw.Draw(image)
        for (r, i) in zip(data, range(0, len(data)*width, width)):
            color = (r >= upper) and above_color or below_color
            if r < 0: y_coord = image.size[1] - (r - dmin) / (float(dmax - dmin + 1) / (height - gap))
            else:     y_coord = image.size[1] - (r - dmin) / (float(dmax - dmin + 1) / (height - gap))
            if longlines:
                draw.rectangle((i, zero, i+width-2, y_coord), fill=color)
            else:
                draw.rectangle((i, y_coord - gap, i+width-2, y_coord), fill=color)
        del draw                                                      

        if output:
            image.save(output, "PNG")
            return output
        else:
            return image

    def draw(self):
        pass



def sparkline_discrete(data, output=None, 
        dmin=None, dmax=None, upper=None, width=2, height=14, 
        below_color='gray', above_color='red', longlines=False):
    """ The source data is a list of values between
      0 and 100 (or 'limits' if given). Values greater than 95 
      (or 'upper' if given) are displayed in red, otherwise 
      they are displayed in green"""
    gap = 4
    if longlines:
        gap = 0
 
    if dmin is None:  dmin = min(data)
    if dmax is None:  dmax = max(data)
    # defaults to the mean
    if upper is None: upper = sum(data) / len(data)

    if dmax < dmin:
        dmax = dmin

    image = Image.new("RGB", (len(data)*width-1, height), 'white') 
    zero = image.size[1] - 1
    if dmin < 0 and dmax > 0:
        zero = image.size[1] - (0 - dmin) / (float(dmax - dmin + 1) / (height - gap))

    draw = ImageDraw.Draw(image)
    for (r, i) in zip(data, range(0, len(data)*width, width)):
        color = (r >= upper) and above_color or below_color
        if r < 0:
            y_coord = image.size[1] - (r - dmin) / (float(dmax - dmin + 1) / (height - gap))
        else:
            y_coord = image.size[1] - (r - dmin) / (float(dmax - dmin + 1) / (height - gap))
        if longlines:
            draw.rectangle((i, zero, i+width-2, y_coord), fill=color)
        else:
            draw.rectangle((i, y_coord - gap, i+width-2, y_coord), fill=color)
    del draw                                                      

    if output:
        image.save(output, "PNG")
        return output
    else:
        return image

def sparkline_smooth(results, output=None, 
        dmin=None, dmax=None, step=2, height=20, \
        min_color='#0000FF', max_color='#00FF00', last_color='#FF0000', \
        has_min=False, has_max=False, has_last=False):
    if dmin is None: dmin = min(results)
    if dmax is None: dmax = max(results)

    image = Image.new("RGB", ((len(results)-1)*step+4, height), 'white')
    draw = ImageDraw.Draw(image)
    coords = zip(range(1,len(results)*step+1, step),
                 [height - 3  - (y-dmin)/(float(dmax - dmin +1)/(height-4)) for y in results])
    draw.line(coords, fill="#888888")
    if has_min == True:
      min_pt = coords[results.index(min(results))]
      draw.rectangle([min_pt[0]-1, min_pt[1]-1, min_pt[0]+1, min_pt[1]+1], fill=min_color)
    if has_max == True:
      max_pt = coords[results.index(max(results))]
      draw.rectangle([max_pt[0]-1, max_pt[1]-1, max_pt[0]+1, max_pt[1]+1], fill=max_color)
    if has_last == True:
      end = coords[-1]
      draw.rectangle([end[0]-1, end[1]-1, end[0]+1, end[1]+1], fill=last_color)
    del draw 
    if output:
        image.save(output, "PNG")
        return output
    else:
        return image

if __name__ == "__main__":
    import random
    #generate sort-of-random list
    d = [x*random.random() for x in [10]*100]
    #sparkline_smooth(d,'/tmp/smooth.png')
    #sparkline_smooth(d).show()
    sparkline_discrete(d).show()
    #sparkline_discrete(d,'/tmp/discrete.png')
    #print " Take a look at /tmp/smooth.png and /tmp/discrete.png"
    
