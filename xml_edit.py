import xml.etree.ElementTree as ET

def xml_generate(
    path: str,
    folder: str,
    filename: str,
    width, height,
    xmin, ymin, xmax, ymax,
    miss: bool
):

    string = ''
    with open('sample/0.txt', 'r') as txtfile:
        string = txtfile.read()

    filepath = f'{folder}\{filename}'
    width, height = (640, 480)
    name = folder
    string = string.format(folder,
                           filename,
                           filepath,
                           width, height,
                           name,
                           xmin, ymin, xmax, ymax)

    if miss:
        with open('miss/miss_list.txt', 'a') as xml_file:
            xml_file.write(filepath+'\n')
        print(f'xml {filename} miss')
    else:
        with open(f'{path}/{filename.split(".")[0]}.xml', 'w') as xml_file:
            xml_file.write(string)
        print(f'xml {filename} completed')
