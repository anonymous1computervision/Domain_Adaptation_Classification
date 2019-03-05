import os


def getInformation(file):
    epoch_list = []
    tr_acc_list = []
    tr_loss_list = []
    src_tst_acc_list = []
    src_tst_fscore_list = []
    tar_tst_acc_list = []
    tar_tst_fscore_list = []

    summarized_list = []

    with open(file, 'r') as f:
        lines = f.readlines()
        for l in range(len(lines)):
            _line_information = lines[l].lower()

            if _line_information.startswith('epoch:'):
                _line_split = _line_information.split(',')

                epoch_list.append(int(_line_split[0][_line_split[0].find('[') + 1:_line_split[0].rfind(']')]))
                tr_acc_list.append(float(_line_split[1][_line_split[1].find('[') + 1:_line_split[1].rfind(']')]))
                tr_loss_list.append(float(_line_split[2][_line_split[2].find('[') + 1:_line_split[2].rfind(']')]))

            if _line_information.startswith('mode: source'):
                _src_tst_acc_info = lines[l + 1].lower()
                _src_tst_fscore_info = lines[l + 14].lower()

                _src_tst_acc = float(_src_tst_acc_info.split(':')[1][1:])
                _src_tst_fscore = float(_src_tst_fscore_info.split(':')[1][1:])

                src_tst_acc_list.append(_src_tst_acc)
                src_tst_fscore_list.append(_src_tst_fscore)

            if _line_information.startswith('mode: target'):
                _tar_tst_acc_info = lines[l + 1].lower()
                _tar_tst_fscore_info = lines[l + 14].lower()

                _tar_tst_acc = float(_tar_tst_acc_info.split(':')[1][1:])
                _tar_tst_fscore = float(_tar_tst_fscore_info.split(':')[1][1:])

                tar_tst_acc_list.append(_tar_tst_acc)
                tar_tst_fscore_list.append(_tar_tst_fscore)

        for e, t_a, t_l, s_t_a, s_t_f, t_t_a, t_t_f in zip(epoch_list, tr_acc_list, tr_loss_list, src_tst_acc_list,
                                                           src_tst_fscore_list, tar_tst_acc_list, tar_tst_fscore_list):
            summarized_list.append([e, t_a, t_l, s_t_a, s_t_f, t_t_a, t_t_f])

    return summarized_list


def sortList(summarized_list):
    sorted_list = sorted(summarized_list, key=lambda s: s[-2], reverse=True)

    return sorted_list


def showResults(sorted_list, count):
    for i in range(count):
        # print(
        #     'Epoch [{}], Training Accuracy [{}], Validation Accuracy [{}], Training Loss [{}], Validation Loss [{}], '
        #     'Source Test Accuracy [{}], Source Test FScore [{}], Target Test Accuracy [{}], Target Test FScore [{}]'.format(
        #         sorted_list[i][0], sorted_list[i][1], sorted_list[i][2], sorted_list[i][3], sorted_list[i][4],
        #         sorted_list[i][5], sorted_list[i][6], sorted_list[i][7], sorted_list[i][8]))

        print(
            'Epoch [%d], Source Test Accuracy [%.4f], Source F-value [%.4f], Target Test Accuracy [%.4f], '
            'Target F-value [%.4f]' % (
                sorted_list[i][0], sorted_list[i][-4], sorted_list[i][-3], sorted_list[i][-2], sorted_list[i][-1]))
    print()


def main():
    top_k = 20
    file_dir = 'D:/DLD_Classification_DomainAdaptation_Experimental_Results/CycleGAN/Step2/'
    file_name_list = os.listdir(file_dir)
    print(file_name_list)
    for file in file_name_list:
        print('File Name {}'.format(file))
        info_list = getInformation(file_dir + file)
        sort_list = sortList(info_list)
        showResults(sort_list, top_k)


if __name__ == '__main__':
    main()
