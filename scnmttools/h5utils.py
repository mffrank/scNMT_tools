import h5py
import sys

def printH5(h5file, indent=4, path='/', offset=0, target_stream=sys.stdout):
    '''Recursively prints H5 file tree.
    
    Parameters:
    h5file: The h5py.File object
    indent: how many spaces are used to indentate one level (default: 4)
    path: the start path (default: '/')
    offset: how many spaces of indentation to start with (default: 0)
    target_stream: where to print to (default: sys.stdout)
    '''
    spaces = ' ' * (indent * offset)
    for key in h5file[path].keys():
        try:
            item = h5file[path][key]
            print(spaces + str(item), file=target_stream)
            if isinstance(item, h5py.Group):
                printH5(h5file, indent = indent, path = path + key + '/', 
                        offset = offset + 1, target_stream = target_stream)
        except KeyError:
            ''' Broken external lists will show up as a key in the enumerator, 
            but still throw a KeyError when accessed.'''
            print(spaces + '<BROKEN EXTERNAL LINK: "%s/%s">' % (path, key), 
                  file = target_stream)
