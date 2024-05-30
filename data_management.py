import os
import sys
import json
import re

def retain_letters_and_spaces(input_string):
    # 将所有空格字符（包括特殊空格如\xa0）统一替换为普通空格
    normalized_string = re.sub(r'\s+', ' ', input_string)
    # 使用正则表达式保留英文字母和空格
    result = re.sub(r'[^A-Za-z ]', '', normalized_string)
    return result


def manage_data_and_sort_words(file_path):

    word_count_dict={}
    fine_data_list = []
    with open(file_path, 'r', encoding='utf-8') as file_in:
        for one_line in file_in:
            one_line = one_line.strip('\n')
            if len(one_line) == 0:
                continue
            
            one_line = retain_letters_and_spaces(one_line)

            word_list = one_line.split(' ')

            # word count
            new_word_list = []
            for one_word in word_list:

                one_word = one_word.lower()

                if one_word == '':
                    continue
                
                new_word_list.append(one_word)

                if one_word not in word_count_dict.keys():
                    word_count_dict.update({one_word : 1})
                else:
                    word_count_dict[one_word] += 1
    
            fine_data_list.append(new_word_list)

    return word_count_dict, fine_data_list


if __name__ == '__main__':
    # file_path = './data_source/level-0.txt'
    # word_dict_path = './word_dict.json'
    word_count_dict_l0, fine_data_list_l0 = manage_data_and_sort_words('./data_source/level-0.txt')
    word_count_dict_l1, fine_data_list_l1 = manage_data_and_sort_words('./data_source/level-1.txt')
    # word_count_dict_l2, fine_data_list_l2 = manage_data_and_sort_words('./data_source/level-2.txt')

    # save the fine data
    lv = 0
    for one_data_list in [fine_data_list_l0, fine_data_list_l1]:
        one_file_path = './fine_data/fine_data_lv' + str(lv)
        print(one_file_path)
        with open(one_file_path, 'w', encoding='utf-8') as data_out:
            for one_fine_list in one_data_list:
                one_fine_str = " ".join(one_fine_list) + "\n"
                data_out.write(one_fine_str)
        lv +=1
    

    # word count merge
    word_count_dict = {}
    for one_dict in [word_count_dict_l0, word_count_dict_l1]:
        for one_key in one_dict.keys():
            if one_key not in word_count_dict.keys():
                word_count_dict.update({one_key : one_dict[one_key]})
            else:
                word_count_dict[one_key] = one_dict[one_key]

    sorted_words = [word for word, count in sorted(word_count_dict.items(), key=lambda item: item[1], reverse=True)]

    word_to_index_dict = {}
    idx = 1
    for one_word in sorted_words:
        word_to_index_dict.update({one_word : idx})
        idx+=1

    with open('./index_dictionary/idx_dict.json', 'w', encoding='utf-8') as json_out:
        json_obj = json.dumps(word_to_index_dict, indent=4, ensure_ascii=False)
        json_out.write(json_obj)
    
    print(word_to_index_dict)

