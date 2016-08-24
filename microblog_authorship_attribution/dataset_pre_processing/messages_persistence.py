"""
Auxiliary code for manipulating (read and write) Twitter messages in dataset files.
"""


import codecs


def read(filename):
    messages = []
    str_buffer = u''
    with codecs.open(filename, encoding='utf-8', mode='r', buffering=1, errors='strict') as fd:
        read = False
        for line in fd:
            if not read:
                if line[len(line)-2] == u'{':
                    read = True
            else:
                if line == u'}\n':
                    read = False
                    parts = str_buffer.split('#POS')
                    message = {}
                    message['tweet'] = parts[0][:-1]        # strips the last \n character
                    message['pos'] = parts[1] if len(parts) > 1 else None
                    message['full'] = str_buffer[:-1]       # strips the last \n character
                    messages.append(message)
                    str_buffer = u''
                else:
                    str_buffer += line
    return messages


def write(messages, id, filename):
    with codecs.open(filename, encoding='utf-8', mode='w+', errors='strict') as fd:
        for message in messages:
            fd.write(u'\n'.join([u'{', message[id], u'}\n']))
