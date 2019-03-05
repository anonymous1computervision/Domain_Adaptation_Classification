import sys

sys.path.append('../Data_Initialization/')
import numpy as np
import Initialization as init


def f_value(matrix, num_class):
    f = 0.0
    length = len(matrix[0])
    for i in range(length):
        recall = matrix[i][i] / np.sum([matrix[i][m] for m in range(num_class)])
        precision = matrix[i][i] / np.sum([matrix[n][i] for n in range(num_class)])
        result = (recall * precision) / (recall + precision)
        f += result
    f *= (2 / num_class)

    return f


def validation_procedure(validation_data, distribution_op, loss_op, inputX, inputY, num_class, batch_size, is_training,
                         session):
    confusion_matrics = np.zeros([num_class, num_class], dtype="int")
    val_loss = 0.0

    val_batch_num = int(np.ceil(validation_data[0].shape[0] / batch_size))
    for step in range(val_batch_num):
        _validationImg = validation_data[0][step * batch_size:step * batch_size + batch_size]
        _validationLab = validation_data[1][step * batch_size:step * batch_size + batch_size]

        [matrix_row, matrix_col], tmp_loss = session.run([distribution_op, loss_op],
                                                         feed_dict={inputX: _validationImg,
                                                                    inputY: _validationLab,
                                                                    is_training: False})
        for m, n in zip(matrix_row, matrix_col):
            confusion_matrics[m][n] += 1

        val_loss += tmp_loss

    validation_accuracy = float(np.sum([confusion_matrics[q][q] for q in range(num_class)])) / float(
        np.sum(confusion_matrics))
    validation_loss = val_loss / val_batch_num

    return validation_accuracy, validation_loss


def test_procedure(test_data, distribution_op, inputX, inputY, mode, num_class, batch_size, session, is_training,
                   ckptDir, model):
    confusion_matrics = np.zeros([num_class, num_class], dtype="int")

    tst_batch_num = int(np.ceil(test_data[0].shape[0] / batch_size))
    for step in range(tst_batch_num):
        _testImg = test_data[0][step * batch_size:step * batch_size + batch_size]
        _testLab = test_data[1][step * batch_size:step * batch_size + batch_size]

        matrix_row, matrix_col = session.run(distribution_op, feed_dict={inputX: _testImg,
                                                                         inputY: _testLab,
                                                                         is_training: False})
        for m, n in zip(matrix_row, matrix_col):
            confusion_matrics[m][n] += 1

    test_accuracy = float(np.sum([confusion_matrics[q][q] for q in range(num_class)])) / float(
        np.sum(confusion_matrics))
    detail_test_accuracy = [confusion_matrics[i][i] / np.sum(confusion_matrics[i]) for i in
                            range(num_class)]
    log0 = "Mode: " + mode
    log1 = "Test Accuracy : %g" % test_accuracy
    log2 = np.array(confusion_matrics.tolist())
    log3 = ''
    for j in range(num_class):
        log3 += 'category %s test accuracy : %g\n' % (init.pulmonary_category[j], detail_test_accuracy[j])
    log3 = log3[:-1]
    log4 = 'F_Value : %g\n' % f_value(confusion_matrics, num_class)

    init.save2file(log0, ckptDir, model)
    init.save2file(log1, ckptDir, model)
    init.save2file(log2, ckptDir, model)
    init.save2file(log3, ckptDir, model)
    init.save2file(log4, ckptDir, model)


def test_procedure_DA(test_data, distribution_op, inputX, inputY, mode, num_class, batch_size, session, is_training,
                      keep_rate, ckptDir, model):
    confusion_matrics = np.zeros([num_class, num_class], dtype="int")

    tst_batch_num = int(np.ceil(test_data[0].shape[0] / batch_size))
    for step in range(tst_batch_num):
        _testImg = test_data[0][step * batch_size:step * batch_size + batch_size]
        _testLab = test_data[1][step * batch_size:step * batch_size + batch_size]

        matrix_row, matrix_col = session.run(distribution_op, feed_dict={inputX: _testImg,
                                                                         inputY: _testLab,
                                                                         is_training: False,
                                                                         keep_rate: 0.5})
        for m, n in zip(matrix_row, matrix_col):
            confusion_matrics[m][n] += 1

    test_accuracy = float(np.sum([confusion_matrics[q][q] for q in range(num_class)])) / float(
        np.sum(confusion_matrics))
    detail_test_accuracy = [confusion_matrics[i][i] / np.sum(confusion_matrics[i]) for i in
                            range(num_class)]
    log0 = "Mode: " + mode
    log1 = "Test Accuracy : %g" % test_accuracy
    log2 = np.array(confusion_matrics.tolist())
    log3 = ''
    for j in range(num_class):
        log3 += 'category %s test accuracy : %g\n' % (init.pulmonary_category[j], detail_test_accuracy[j])
    log3 = log3[:-1]
    log4 = 'F_Value : %g\n' % f_value(confusion_matrics, num_class)

    init.save2file(log0, ckptDir, model)
    init.save2file(log1, ckptDir, model)
    init.save2file(log2, ckptDir, model)
    init.save2file(log3, ckptDir, model)
    init.save2file(log4, ckptDir, model)
