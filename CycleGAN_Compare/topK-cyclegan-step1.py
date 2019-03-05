import copy
import os


def getInformation(file):
    epoch_list = []
    g_loss_list = []
    f_loss_list = []
    dx_loss_list = []
    dy_loss_list = []
    cycle_loss_list = []

    summarized_list = []

    with open(file, 'r') as f:
        lines = f.readlines()
        for l in range(len(lines)):
            _line_information = lines[l].lower()

            if _line_information.startswith('epoch:'):
                _line_split = _line_information.split(',')

                epoch_list.append(int(_line_split[0][_line_split[0].find('[') + 1:_line_split[0].rfind(']')]))
                g_loss_list.append(float(_line_split[1][_line_split[1].find('[') + 1:_line_split[1].rfind(']')]))
                f_loss_list.append(float(_line_split[2][_line_split[2].find('[') + 1:_line_split[2].rfind(']')]))
                dx_loss_list.append(float(_line_split[3][_line_split[3].find('[') + 1:_line_split[3].rfind(']')]))
                dy_loss_list.append(float(_line_split[4][_line_split[4].find('[') + 1:_line_split[4].rfind(']')]))
                cycle_loss_list.append(float(_line_split[5][_line_split[5].find('[') + 1:_line_split[5].rfind(']')]))

        for e, g, f, dx, dy, cyc in zip(epoch_list, g_loss_list, f_loss_list, dx_loss_list, dy_loss_list,
                                        cycle_loss_list):
            summarized_list.append([e, g, f, dx, dy, cyc])

    return summarized_list


def sortList(summarized_list):
    sorted_list = sorted(summarized_list, key=lambda s: s[-1])

    return sorted_list


def showResults(sorted_list, count):
    for i in range(count):
        # print(
        #     'Epoch [{}], Training Accuracy [{}], Validation Accuracy [{}], Training Loss [{}], Validation Loss [{}], '
        #     'Source Test Accuracy [{}], Source Test FScore [{}], Target Test Accuracy [{}], Target Test FScore [{}]'.format(
        #         sorted_list[i][0], sorted_list[i][1], sorted_list[i][2], sorted_list[i][3], sorted_list[i][4],
        #         sorted_list[i][5], sorted_list[i][6], sorted_list[i][7], sorted_list[i][8]))

        print(
            'Epoch [%d], G loss [%.4f], F loss [%.4f], DX loss [%.4f], DY loss [%.4f], Cycle loss [%.4f]' % (
                sorted_list[i][0], sorted_list[i][1], sorted_list[i][2], sorted_list[i][3], sorted_list[i][4],
                sorted_list[i][5]))
    print()


def main():
    top_k = 20
    file_dir = 'D:/DLD_Classification_DomainAdaptation_Experimental_Results/CycleGAN/Step1/'
    file_name_list = os.listdir(file_dir)
    print(file_name_list)
    for file in file_name_list:
        print('File Name {}'.format(file))
        info_list = getInformation(file_dir + file)
        sort_list = sortList(info_list)
        showResults(sort_list, top_k)


if __name__ == '__main__':
    main()
