from hachoir.parser import createParser
from hachoir.metadata import extractMetadata
import sys

def creation_date(filepath):
    parser = createParser(filepath)
    metadata = extractMetadata(parser)

    if not metadata:
        return None

    # Check for creation date keys, in order of most likely to be correct
    # note: can get metadata keys like so:
    # print(metadata.exportPlaintext(human=False))

    if metadata.has('date_time_original'):
        return metadata.get('date_time_original')
    if metadata.has('creation_date'):
        return metadata.get('creation_date')
    return None

print(creation_date(sys.argv[1]))