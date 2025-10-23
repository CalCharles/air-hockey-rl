import os
def custom_xml_path_completion(xml_path):
    """
    Takes in a local xml path and returns a full path.
        if @xml_path is absolute, do nothing
        if @xml_path is not absolute, load xml that is shipped by the package

    Args:
        xml_path (str): local xml path

    Returns:
        str: Full (absolute) xml path
    """
    from airhockey import ASSETS_ROOT
    if xml_path.startswith("/"):
        full_path = xml_path
    else:
        full_path = os.path.join(ASSETS_ROOT, xml_path)
    return full_path