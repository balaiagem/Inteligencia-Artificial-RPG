    from os import close
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score
    from sklearn import datasets
    from sklearn import svm
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    from pybrain3.tools.shortcuts import buildNetwork
    from pybrain3.datasets import SupervisedDataSet
    from pybrain3.supervised.trainers import BackpropTrainer
    from pybrain3.structure.modules import SoftmaxLayer
    from pybrain3.structure.modules import SigmoidLayer
    from pybrain3.tools.validation import CrossValidator, ModuleValidator
    import matplotlib.pyplot as plt
    from datetime import datetime

    arraydeerros = []
    arraydetentativa = []
    rede = buildNetwork(69, 100, 100, 5)
    base = SupervisedDataSet(69, 5) 
    print(rede)

    arquivo = open('base.txt', 'r')
    arquivo.seek(0, 0)
    for linha in arquivo.readlines():
        print(linha)
        l = [float(x) for x in linha.strip().split(',') if x != '']
        indata = l[:69]
        outdata = l[69:]

        print(indata)
        print(outdata)

        base.addSample(indata, outdata)

    treinamento = BackpropTrainer(
        rede, dataset=base, learningrate=0.01, momentum=0.5)
    
    logs = open('logs.txt','w')
    logs.write("-------------TESTES REALIZADOS EM "+str(datetime.today())+"-------------\n")
    error = 2
    iteration = 0 
    outputs=[]
    while error > 0.001:
       error = treinamento.train()
       outputs.append(error)
       iteration += 1
       print(iteration,error)
       logs.write("TENTATIVA " +str(iteration) +": " +str(error) +" de erro.\n")
       arraydeerros.append(error)
       arraydetentativa.append(iteration)
    
    logs.write("--------------------------FIM-------------------------\n")
    logs.close()
    plt.plot(arraydetentativa,arraydeerros)
    plt.title('Gráfico de aprendizado de máquina')
    plt.xlabel('Nº da Tentativa')
    plt.ylabel('Valor do Erro')
    plt.savefig('grafico.jpg', format='jpg')
    plt.show()
    

    rede.activate([1,6,2,6,2,27,18,17,11,2,7,1,0,3,3,0,20,13,15,13,13,12,5,1,1,1,1,1,1,1,1,1,1,7,1,2,1,1,1,1,1,1,1,1,1,1,9,1,1,1,1,7,5,1,1,1,1,1,1,6,1,1,1,1,9,1,1,1,1])
