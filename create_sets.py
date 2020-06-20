files = ['Sonar_all_data.txt','Dragonfly_all_data.txt',
         'tensorflow_all_data.txt','devilution_all_data.txt',
         'flutter_all_data.txt',
         'react_all_data.txt','spritejs_all_data.txt']

for label,file in enumerate(files):
    lines=0
    with open(file) as f:
        for l in f:
            if  len(l.strip()) > 0:
                lines += 1
    val_size = round(0.15 * lines)
    train_size = round(0.7 * lines)
    print(train_size)
    with open(file) as f:
        # with open(str(label)+"_train.txt",'a') as w:
        #     i = 0
        #     while i < train_size:
        #         l = f.readline()
        #         if  len(l.strip()) > 0:
        #             w.write(l)
        #             i += 1
        # with open(str(label)+"_val.txt",'a') as w:
        #     i = 0
        #     while i < val_size:
        #         l = f.readline()
        #         if  len(l.strip()) > 0:
        #             w.write(l)
        #             i += 1
        #
        # with open(str(label)+"_test.txt",'a') as w:
        #     i = 0
        #     for l in f:
        #         if  len(l.strip()) > 0:
        #             w.write(l)
        #             i += 1
        with open(str(label)+"_train.txt",'a') as w:
            i = 0
            for l in f:
                if  len(l.strip()) > 0:
                    w.write(l)
                    i += 1