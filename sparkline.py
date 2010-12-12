#!/usr/bin/python

"""
    sparkline.py

        A python module for generating sparklines.
        Requires the Python Imaging Library 

    NOTE:
        this module is a rewrite of Joe Gregorio and Matthew Perry's code, 
        (see http://www.perrygeo.net/wordpress/?p=64)
        hence inherits the license and the version, 
        in a more code style of jquery.sparkline.
        (see http://omnipotent.net/jquery.sparkline/)

"""

__author__ = "Janghwan Kim (janghwan@gmail.com)"
__copyright__ = "Copyright 2010, Janghwan Kim"
__contributors__ = ['Joe Gregorio', 'Matthew Perry', 'Alan Powell']
__version__ = "0.2"
__license__ = "MIT"

import Image, ImageDraw
from collections import Sequence
import re


class Canvas:
    def __init__(self, width=None, height=None):
       self.image = None
       if width is not None and height is not None:
            self.image = Image.new("RGB", (width, height), 'white')

    def draw(self, data, **options):
        # set default common options
        opt = defaults.common.copy().merge(**options)

        if self.image is not None:
            if 'width'  not in options: opt['width']  = self.image.size[0]
            if 'height' not in options: opt['height'] = self.image.size[1]

        chart = self.find_chart(data, opt)
        opt = chart.resolve_options(data, opt)

        # determine canvas size and resolve options
        if self.image is None:
            canvas_size = chart.get_canvas_size(data, opt)
            self.image = Image.new("RGB", canvas_size, 'white')

        # now draw
        chart.draw(data, self.image, opt)
        return self

    def find_chart(self, data, opt):
        chart_name = opt['type']
        if chart_name not in globals():
            if not chart_name.endswith("Chart"):
                chart_name = chart_name.title() + "Chart"

        chart = globals().get(chart_name, defaults.common.type)(data, opt)
        return chart

    def save(self, output, ext="PNG"):
        self.image.save(output, ext)
        return output

    def show(self):
        return self.image.show()

    def __repr__(self):
        module = self.__module__
        class_name = self.__class__.__name__
        size = "x".join(map(str, self.image.size))
        _id = id(self)
        return "<%(module)s.%(class_name)s mode=RGB size=%(size)s at 0x%(_id)x>" % locals()


class Options(dict):
    __ACCEPTABLE__ = re.compile("^[_a-zA-Z][_a-zA-Z0-9]*")
    def __init__(self, **_dict):
        self.merge(**_dict)

    def __setitem__(self, key, value):
        if key not in self.__class__.__RESERVED_KEYS__:
            if isinstance(key, basestring):
                if self.__class__.__ACCEPTABLE__.match(key):
                    setattr(self, key, value)
        return super(Options, self).__setitem__(key, value)

    def setdefault(self, k, d=None):
        if k not in self:
            self[k] = d
        return self[k]

    def merge(self, **_dict):
        for k,v in _dict.copy().iteritems():
            self[k] = v 
        return self

    def copy(self):
        return Options(**super(Options, self).copy())
Options.__RESERVED_KEYS__ = frozenset(dir(Options))
        

class defaults:
    common = Options(**{
        'type'  : 'LineChart',
        'width' : None,
        'height': None,
        'line_color': "#888888",
        'fill_color': False,
        'chart_range_min': None,
        'chart_range_max': None,
        #'composite': None, #?
    })

    line = Options(**{
        'default_pixels_per_value': None,
        'spot_color': '#f80',
        'min_spot_color': '#f80',
        'max_spot_color': '#f80',
        'spot_radius': int(round(1.5)),
        'line_width': 1,
        'normal_range_min': None, 
        'normal_range_max': None, 
        'normal_range_color': '#ccc', 
        'xvalues': None,
        'chart_range_clip':   False, #?
        'chart_range_clip_x': False, #?
        'chart_range_min_x': None, #?
        'chart_range_max_x': None, #?
    })

    bar = Options(**{
        'bar_color': '#00f',     # Colour used for postive values
        'neg_bar_color': '#f44', # Colour used for negative values
        'zero_color': None,    # Colour used for values equal to zero
        #'null_color': None,    #?
        'bar_width': 4,        # Width of each bar, in pixels
        'bar_spacing': 1,      # Space between each bar, in pixels
        'zero_axis': None,     # Centers the y-axis at zero if true (default is to automatically do the right thing)
        'color_map': None,     # Map override colors to certain values 
    })



class BaseChart(object):
    def __init__(self, *args, **kwargs): pass

    def resolve_options(self, data, opt, ignore=[]):
        should_update_option = lambda opt_name: \
            getattr(opt, opt_name, False) is None and opt_name not in ignore
        # data may be [3,5,2] or [(1,3), (2,5), (3,2)]
        if len(data) > 0 and isinstance(data[0], Sequence):
            data = [y for (x,y) in data]

        if should_update_option('chart_range_min'): 
            opt['chart_range_min'] = int(self.min(data))
        if should_update_option('chart_range_max'): 
            opt['chart_range_max'] = int(max(data))

        if should_update_option('height'): 
            opt['height'] = opt.chart_range_max - opt.chart_range_min

        return opt

    def draw(self):
        raise NotImplemented("implement this is child class")

    def get_canvas_size(self, data, options):
        raise NotImplemented("implement this is child class")

    def min(self, data):
        return min(y for y in data if y is not None)
    
    def compute_x(self, x, opt):
        raise NotImplemented("implement this is child class")
    def compute_y(self, y, opt):
        raise NotImplemented("implement this is child class")



class LineChart(BaseChart):
    def resolve_options(self, data, opt):
        for key, default in defaults.line.items():
            if key not in opt:
                opt[key] = default

        if len(data) > 0 and isinstance(data[0], Sequence):
            data = [y for (x,y) in data]
            opt['xvalues'] = [x for (x,y) in data]

        # default pixel per value
        if opt.default_pixels_per_value is None:
            if opt.width: opt['default_pixels_per_value'] = float(opt.width - 4) / (len(data)-1)
            else:         opt['default_pixels_per_value'] = 3

        # min/max spot color
        if opt.max_spot_color is True: opt['max_spot_color'] = opt.spot_color
        if opt.min_spot_color is True: opt['min_spot_color'] = opt.spot_color

        # normal range min/max
        if opt.normal_range_min is None: opt['normal_range_min'] = self.min(data)
        if opt.normal_range_max is None: opt['normal_range_max'] = max(data)

        # resolve common options
        opt = super(LineChart, self).resolve_options(data, opt)

        return opt


    def compute_x(self, x, opt):
        return 1 + x * opt.default_pixels_per_value

    def compute_y(self, y, opt):
        scale = float(opt.chart_range_max - opt.chart_range_min + 1)/(opt.height - 4)
        return opt.height - 3  - (y - opt.chart_range_min) / scale

    def get_coords(self, data, opt):
        ''' compute coords, removing any None data '''
        xcoords = opt['xvalues'] or range(len(data))
        xcoords = [self.compute_x(x, opt) for x in xcoords]
        ycoords = [self.compute_y(y, opt) if y is not None else None for y in data]
        return zip(xcoords, ycoords)

    def get_canvas_size(self, data, opt):
        width  = (len(data)-1) * opt.default_pixels_per_value + 4
        return (width, opt.height)

    def draw(self, data, image, opt):
        coords = self.get_coords(data, opt)
        draw = ImageDraw.Draw(image)
        if draw:
            self._fill_normal_range(draw, coords, opt)
            self._fill_color(draw, coords, opt)
            self._draw_line(draw, coords, opt)
            self._draw_spots(draw, data, coords, opt)
        del draw

        return image


    def _fill_normal_range(self, _draw, coords, opt):
        if opt.normal_range_min and opt.normal_range_max and opt.normal_range_color:
            y1 = self.compute_y(opt.normal_range_min, opt)
            y2 = self.compute_y(opt.normal_range_max, opt)
            normal_range = [coords[0][0],y1, coords[-1][0],y2]
            _draw.rectangle(normal_range, fill=opt.normal_range_color)
        

    def _draw_spots(self, _draw, data, coords, opt):
        ''' last spot / min spot / max spot '''
        draw_spot = lambda _draw, pt, color: \
            _draw.ellipse([pt[0]-opt.spot_radius, pt[1]-opt.spot_radius, pt[0]+opt.spot_radius, pt[1]+opt.spot_radius], fill=color)
        if opt.min_spot_color:
            draw_spot(_draw, coords[data.index(self.min(data))], opt.min_spot_color)
        if opt.max_spot_color:
            draw_spot(_draw, coords[data.index(max(data))], opt.max_spot_color)
        if opt.spot_color:
            last_xy = max(((x,y) for (x,y) in coords if y is not None), key=lambda (x,y): x)
            draw_spot(_draw, last_xy, opt.spot_color)

    def _draw_except_void(self, _draw, coords, draw_func):
        ''' draw except the None area '''
        start = 0
        for end in (i for i,(x,y) in enumerate(coords) if y is None):
            draw_func(_draw, coords[start:end])
            start = end+1
        if coords[start:]:
            draw_func(_draw, coords[start:])

    def _fill_color(self, _draw, coords, opt):
        if opt.fill_color:
            draw_poly = lambda _draw, xy: _draw.polygon([(xy[0][0],opt.chart_range_max)] + xy + [(xy[-1][0],opt.chart_range_max)], fill=opt.fill_color)
            self._draw_except_void(_draw, coords, draw_poly)

    def _draw_line(self, _draw, coords, opt):
        draw_line = lambda _draw, coords: _draw.line(coords, fill=opt.line_color, width=opt.line_width)
        self._draw_except_void(_draw, coords, draw_line)
        


class BarChart(BaseChart):

    def __init__(self, data, opt):
        if self.axis_is_at_bottom(data, opt):
                self.chart = BarChartAbs(data, opt)
        else:   self.chart = BarChartMiddle(data, opt)

    def axis_is_at_bottom(self, data, opt):
        if opt.get('zero_axis') is not None:
            return opt.zero_axis

        if self.min(data) < 0:
            return False

        return True

    def resolve_options(self, data, opt):
        # set names
        for key in defaults.bar.keys():
            opt.setdefault(key, None)

        # compute bar_width from width if given
        if opt.width is not None and opt.bar_width is None:
            # bar_spacing must be resolved first. (default if not given)
            default_bar_spacing = defaults.bar.bar_spacing
            opt['bar_spacing'] = opt.bar_spacing or default_bar_spacing
            # 
            data_length = len(data) or 1
            opt['bar_width'] = (opt.width + opt.bar_spacing) / data_length - 1 - opt.bar_spacing 

        # set default
        for key, default in defaults.bar.items():
            if opt.get(key) is None:
                opt[key] = default

        # zero axis
        min_data = self.chart.min(data)

        # chart_range_min
        if opt.chart_range_min is None: 
            opt['chart_range_min'] = int(min_data)

        # resolve common options
        opt = super(BarChart, self).resolve_options(data, opt)

        # adjust height
        if min_data <= 0:
            opt['height'] += 1

        return opt

    def get_canvas_size(self, data, opt):
        width = len(data) * (opt.bar_width + opt.bar_spacing + 1) - opt.bar_spacing
        return (width, opt.height)

    def compute_x(self, i, opt):
        return i * (opt.bar_width + opt.bar_spacing + 1)

    def compute_y(self, y, opt):
        raise NotImplemented

    def compute_axis(self, y, opt):
        raise NotImplemented


    def select_color(self, i, y, opt):
        # 
        if y >= 0:  color = opt.bar_color
        else:       color = opt.neg_bar_color

        # zero color
        if y is 0 and opt.zero_color is not None:
            color = opt.zero_color

        ## null color
        #if y is None and opt.null_color is not None:
        #    color = opt.null_color

        # color map
        if opt.color_map is not None:
            # {-1: 'red', 0: 'green', 1: 'blue'}
            if isinstance(opt.color_map, dict):
                color = opt.color_map.get(y, color)
            # ['red', 'green', 'blue']
            elif isinstance(opt.color_map, Sequence):
                if i < len(opt.color_map):
                    color = opt.color_map[i]

        return color


    def draw(self, data, image, opt):
        draw = ImageDraw.Draw(image)
        if draw:
            _zero = self.chart.compute_axis(opt)
            for i,y in enumerate(data):
                if y is None:
                    continue

                _x = int(self.compute_x(i, opt))
                _y = int(self.chart.compute_y(y, opt))
                color = self.select_color(i,y, opt)
                draw.rectangle([_x,_y, _x + opt.bar_width,_zero], fill=color)
        del draw

        return image



class BarChartMiddle(BarChart):
    def __init__(self, *args, **kwargs): pass

    def compute_y(self, y, opt):
        scale = float(opt.chart_range_max - opt.chart_range_min + 1)/opt.height
        _y = opt.height/2  - y / scale
        if y < 0: _y += 1
        if y > 0: _y -= 1
        return int(_y)

    def compute_axis(self, opt):
        return self.compute_y(0, opt)


class BarChartAbs(BarChart):
    def __init__(self, *args, **kwargs): pass

    def min(self, data):
        return min(abs(y) for y in data if y is not None)

    def compute_y(self, y, opt):
        scale = float(opt.chart_range_max - opt.chart_range_min + 1)/(opt.height)
        _y = opt.height - (abs(y) - opt.chart_range_min) / scale - 1
        return int(_y)

    def compute_axis(self, opt):
        return opt.height
        #return self.compute_y(0, opt) + 1



def draw(data, **options):
    ''' convenient function. calls sparkline.draw([1,2,3]) '''
    canvas = Canvas()
    return canvas.draw(data, **options)


if __name__ == "__main__":
    import random
    #generate sort-of-random list
    d = [x*(random.random() - 0.2) for x in [10]*50]
    #draw(d).save('/tmp/smooth.png')
    #draw(d).show()
    draw(d, type='bar', bar_width=2, height=20, zero_color="#000", zero_axis=True).show()
    #draw(d, type='bar').save('/tmp/discrete.png')
    #print " Take a look at /tmp/smooth.png and /tmp/discrete.png"
    

