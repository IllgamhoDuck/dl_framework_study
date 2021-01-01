import os
import subprocess
import graphviz
import urllib.request


### Visualizer

def _dot_var(v, verbose=False):
    dot_var = '{} [label="{}", color=orange, style=filled]\n'

    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)
    return dot_var.format(id(v), name)

def _dot_func(f):
    dot_func = '{} [label={}, color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)

    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        # y is weakref so we should use y()
        txt += dot_edge.format(id(f), id(y()))
    return txt

def get_dot_graph(output, verbose=True):
    txt = ''
    funcs = []
    seen_var = set()
    seen_func = set()

    def add_func(f):
        if f not in seen_func:
            funcs.append(f)
            seen_func.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            if x not in seen_var:
                txt += _dot_var(x, verbose)
                seen_var.add(x)

                if x.creator is not None:
                    add_func(x.creator)

    return 'digraph g{\n' + txt + '}'

def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    dot_graph = get_dot_graph(output, verbose)

    # save dot to file
    tmp_dir = os.path.join(os.path.expanduser('~'), '.dezero')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')

    with open(graph_path, 'w') as f:
        f.write(dot_graph)

    # run dot command
    extension = os.path.splitext(to_file)[1][1:]
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True)

    # visualize when using jupyter notebook
    try:
        from IPython import display
        return display.Image(filename=to_file)
    except:
        pass

def visualize_graph(output, verbose=True):
    dot_graph = get_dot_graph(output, verbose)
    return graphviz.Source(dot_graph)


### Utility function for numpy

def reshape_sum_backward(gy, x_shape, axis, keepdims):
    """Reshape gradient appropriately for dezero.functions.sum's backward.

    Args:
        gy (dezero.Variable): Gradient variable from the output by backprop
        x_shape (tuple): Shape used at sum function's forward.
        axis (None or int or tuple of ints): Axis used at sum function's forward.
        keepdims (bool): keepdims used at function's forward.

    Returns:
        dezero.Variable: Gradient variable which is reshaped appropriately

    # If we keep dim this is not necessary
    # But if we don't keep dim and sum with axis there is a problem

    -> if x shape is (2, 3, 4, 7, 5) and we sum with axis (1, 3)
       the result will shown out with shape (2, 4, 5) without keepdim
       for the backpropagation broadcast we should make this to (2, 1, 4, 1, 5)

    -> so when we we say axis (1, 3) is actual axis and gy.shape is (2, 4, 5)
       we use actual axis as index to put 1 at gy.shape

       for a in sorted(actual_axis):
           shape.insert(a, 1)

        make (2, 4, 5) to (2, 1, 4, 1, 5)

    """

    ndim = len(x_shape)
    tupled_axis = axis

    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):
        # if ndim is 3 and axis is -1 then actual axis is 2
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]

        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = gy.shape

    gy = gy.reshape(shape)
    return gy

def sum_to(x, shape):
    """Sum elements along axes to output an array of a given shape.

    Args:
        x (ndarray): Input array.
        shape: shape for the output array

    Returns:
        ndarray: Output array of the shape

    `sum_to` function usage is quite tricky.

    - leading could only followed by case of [[[1, 2, 3]]] so `[1, 2, 3] the data` and `[[]] is leading`
    - to sum we should indicate where to sum directly!
        - when there is (2, 3, 5)
        - we should input shape as (2, 3, 1) to sum the last axis which have shape 5!

    -> if x shape is (1, 1, 5, 2, 7) and shape is (5, 2, 1)
        ndim = 3
        lead = 2
        lead_axis = (0, 1)

        axis = (4,)
        lead_axis + axis = (0, 1, 4)
        y = x.sum((0, 1, 4), keepdims=True)

        so the y shape is (5, 2, 1)
    """
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y

def max_backward_shape(x, axis):
    if axis is None:
        axis = range(x.ndim)
    elif isinstance(axis, int):
        axis = (axis,)
    else:
        axis = axis

    shape = [s if ax not in axis else 1 for ax, s in enumerate(x.shape)]

    return shape


### Download

def show_progress(block_num, block_size, total_size):
    bar_template = "\r[{}] {:.2f}%"

    downloaded = block_num * block_size

    p = downloaded / total_size * 100
    i = int(downloaded / total_size * 30)
    if p >= 100.0: p = 100.0
    if i >= 30: i=30

    bar = "#" * i + "." * (30 - i)
    print(bar_template.format(bar, p), end='')

cache_dir = os.path.join(os.path.expanduser('~'), '.dezero')

def get_file(url, file_name=None):
    """Download a file from the `url` if it is not in the cache.

    The file at the `url` is downloaded to the `~/.dezero`

    Args:
        url (str): URL of the file.
        file_name (str): Name of the file.
                         If `None` is specified the original file name is used.

    Returns:
        str: Absolute path to the saved file
    """
    if file_name is None:
        file_name = url[url.rfind('/') + 1:]

    # Check is cache directory exist
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    file_path = os.path.join(cache_dir, file_name)

    if os.path.exists(file_path):
        return file_path

    print("Downloading: " + file_name)
    try:
        urllib.request.urlretrieve(url, file_path, show_progress)
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise

    print(" Done")

    return file_path


### Etc

def pair(x):
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, tuple):
        assert len(x) == 2
        return x
    else:
        raise ValueError
