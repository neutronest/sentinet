#-*- coding:utf-8 -*-


""" =================== file and text process
"""


def get_lines_from_file_useful(filepath):
    """ get lines from file which label1 = label2 and label1 != 2 or NULL

    Parameters:
    -----------
    filepath: the data path in the system
              type: str

    Return:
    -------
    lines_res : all data in the file
            type: str list list
    """
    lines_res = []

    # temp varialbe init
    group_lines = []
    temp_lines = []
    cur_groupid = ""

    useful_flag = True
    with open(filepath) as file_ob:
        print filepath
        next(file_ob)
        for line in file_ob:
            data = line.split("\t")
            group_id = data[1]
            label1 = data[8]
            label2 = data[9]
            valid = data[12]
            lines_res.append(line)
            """
            # check if is in the same group
            if group_id != cur_groupid:
                # not in the samp group
                # store the old group into group_lines
                # create a another group item
                if temp_lines != []:
                    group_lines.append(temp_lines)
                # re-init
                temp_lines = []
                cur_groupid = group_id
                useful_flag = True
            else:
                # check the flag
                if useful_flag == False:
                    continue
                # check the new line
                if (label1 == "NULL" or label2 == "NULL") \
                   or (valid == "NULL" and label1 != label2) \
                   or (label1 == "2"):
                    useful_flag = False
                    # clean the temp_lines
                    temp_lines = []
                    continue
            temp_lines.append(line)
        # add the last group to group_lines
        group_lines.append(temp_lines)
            """
    file_ob.close()

        # get the group_lines of type:str list list
    # lines_res = [text for group in group_lines for text in group]

    return lines_res


def get_line_from_file(filename):
    """

    """
    texts = []
    with open(filename, "r") as file_ob:
        for line in file_ob:
            texts.append(line)
    file_ob.close()
    return texts


def get_text_only_from_lines(lines):

    texts = []
    for line in lines:
        data = line.split("\t")
        texts.append(data[3])
    return texts

def write_to_file(data, file_name):
    """ write the data to the file

    Parameters:
    -----------
    data: just the data, always is a LIST
    file_name: contain file_path and file_name exactly,
               for example: "../data/chinese_data.txt"
               type: str
    Returns:
    --------
    None

    """
    with open(file_name, 'w') as file_ob:
        for data_line in data:
            file_ob.write(data_line + "\n")
    file_ob.close()
    return


if __name__ == "__main__":
     sample_lines = get_lines_from_file_useful("../data/weibo.tsv")
