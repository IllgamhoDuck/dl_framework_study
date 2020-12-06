import os
import subprocess
import graphviz


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
