import operator
import numpy as np
import collections
import json
import matplotlib
from geopy import Point
from geopy.distance import distance
import matplotlib.pyplot as plt
import math
from scipy import stats


def mape_cal(pred, score):
    '''
    regression task evaluation
    :param pred:
    :param score:
    :return:
    '''
    mape=[]
    for i, j in zip(pred,score):
        if j!=0:
            mape.append(100*abs(i-j)/j)  
    return sum(mape)/len(mape)
    
def wifi_time_2_sec(time_str):
    '''
    :param time_str: 2017-03-06T11:28:26.000Z
    :return: seconds of a time string
    '''
    times=time_str.split('T')[1].split('.')[0].split(':')
    return (3600*int(times[0])+60*int(times[1])+int(times[2]))


def time_2_sec_T(time_str):
    '''
    :param time_str: 2017-03-06T11:28:26.000Z
    :return: seconds of a time string
    '''
    times=time_str.split('T')[1].split('.')[0].split(':')
    return (3600*int(times[0])+60*int(times[1])+int(times[2]))

def time_2_sec(time_str):
    '''
    :param time_str: 20170306222829\n
    :return: seconds of a time string
    '''
    tmp = time_str.split('\n')[0]
    return (int(tmp[-2:])+int(tmp[-4:-2])*60+int(tmp[-6:-4])*3600)

def contain_zh(x,y, minx, miny, xstep, ystep):
    '''
    :param x: lng
    :param y: lat
    :param minx: min lng
    :param miny: min lat
    :param xstep: step of lngs
    :param ystep: step of lats
    :return: grid index of coords
    '''
    (x_,y_) = (int((x-minx)/xstep),int((y-miny)/ystep))
    return (x_,y_)

def contain(x, y, ps):
    '''
    :param x: lng
    :param y: lat
    :param ps: a closed list of coords vector
    :return: point in the polygon or not
    '''
    result = False#-1 means not in the polygon
    j = len(ps)-1

    for i in range(0,len(ps)):
        if ((ps[i][1] > y) != (ps[j][1] > y)) and \
                (x < (ps[j][0] - ps[i][0]) * (y - ps[i][1]) / (ps[j][1]-ps[i][1]) + ps[i][0]):
            result = True
            j = i
            i += 1
    return result

def contain_poly(x, y, ps):
    '''
    a better version of contain
    :return: point in the polygon or not
    '''
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon
    poly_ = Polygon(ps)
    pt = Point(x, y)
    return True if poly_.contains(pt) else False


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def cdf_draw(val_list, cut_val=None):
    '''
    used to calculate cdf values of some value list
    :param val_list: value list
    :param cut_val: cut the data by some x-axis threshold
    :return: xvals, yvals for ploting cdf
    '''
    if cut_val==None:
        thold=max(val_list)
    else:
        thold=cut_val
    temp_dict=collections.defaultdict(int)
    for val in val_list:
        if val<=thold:
            temp_dict[val]+=1
    xvals=[];yvals=[];y_temp=0;sum_=sum(temp_dict.values())
    temp_dict_sorted=sorted(temp_dict.items(),key=operator.itemgetter(0))
    for item in temp_dict_sorted:
        xvals.append(item[0])
        y_temp+=item[1]
        yvals.append(100*y_temp/sum_)

    print ("Percentage of remaining data over all data: ", 100*sum(temp_dict.values())/len(val_list))
    return xvals, yvals

def heatmap_data_format(xvals, yvals, xdim=50, ydim=50):
    # TODO: format data for heatmap
    max_xval=max(xvals); min_xval=min(xvals)
    max_yval=max(yvals); min_yval=min(yvals)
    mat_=np.zeros((ydim, xdim))

    for x, y in zip(xvals, yvals):
        try:
            mat_[math.floor(ydim*(y-min_yval)/(max_yval-min_yval)),\
                math.floor(xdim*(x-min_xval)/(max_xval-min_xval))]+=1
        except:
            continue
    return mat_


def data_dist(val_list):
    '''
    return distribution of values
    :param val_list: value list
    :return: dict with val, %
    '''
    val_count=collections.defaultdict(int)
    for val in val_list:
        val_count[val]+=1
    val_dist=collections.defaultdict(float)
    sum_=len(val_list)
    for k, v in val_count.items():
        val_dist[k]=100*v/sum_
    # print (list(val_dist.keys()))
    return val_dist

def error_bar_draw(list_, num=1):
    # TODO: return params for error bar plot, every num
    xvals=np.arange(0, len(list_), num)
    yvals=[sum(list_[i:i+num])/num for i in range(0,len(list_),num)]
    xerr_=0
    yerr1=[sum(list_[i:i+num])/num-min(list_[i:i+num]) for i in range(0,len(list_),num)]
    yerr2=[max(list_[i:i+num])-sum(list_[i:i+num])/num for i in range(0,len(list_),num)]
    yerr_=[yerr1, yerr2]
    return xvals, yvals, xerr_, yerr_


def entropy_cal(obj):
    # TODO: calculate entropy
    prob_=[]
    if type(obj)==list: # a list of repetitive values 
        count_dict=collections.Counter(obj)
        all_=sum(list(count_dict.values()))
        for loc, count_ in count_dict.items():
            prob_.append(count_/all_)
    elif type(obj)==dict: # already a dict
        all_=sum(list(obj.values()))
        for loc, count_ in obj.items():
            prob_.append(count_/all_)
    return stats.entropy(prob_)
    

def rg_cal(raw_trace):
    # TODO: calculate radius of gyration
    # input: list of traces, a trace: list of coordinates [lng, lat]
    traces=[raw_trace]
    lon_sum = 0
    lat_sum = 0
    count = 0
    for i in range(len(traces)):
        # First Location
        lon_sum += traces[i][0][0]
        lat_sum += traces[i][0][1]
        count += 1

        if len(traces[i]) > 1:
            # Last Location
            lon_sum += traces[i][-1][0]
            lat_sum += traces[i][-1][1]
            count += 1

    center_lon = lon_sum / count
    center_lat = lat_sum / count

    distances = []

    for i in range(len(traces)):
        # First Location
        a = Point(latitude=center_lat, longitude=center_lon)
        b = Point(latitude=traces[i][0][1], longitude=traces[i][0][0])
        d = distance(a, b).m
        distances.append(d)

        if len(traces[i]) > 1:
            # Last Location
            a = Point(latitude=center_lat, longitude=center_lon)
            b = Point(latitude=traces[i][-1][1], longitude=traces[i][-1][0])
            d = distance(a, b).m
            distances.append(d)

    rg = math.sqrt(sum([d**2 for d in distances]) / len(distances))

    return rg

