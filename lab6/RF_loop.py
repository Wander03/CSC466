import RFAuthorship
import datetime


def main():
    run = 1
    for i in [50]:
        for j in[500, 1000]:
            for t in [3, 5]:
                for thresh in [.1, .05]:
                    print(datetime.datetime.now())
                    print('Run:', run)
                    print(i, j, t, thresh)
                    RFAuthorship.main(['hehe', str(i), str(j), str(t), str(thresh), '0', f'{i}_{j}_{t}_{thresh}_0'])
                    print(datetime.datetime.now())
                    run += 1

if __name__ == '__main__':
    main()
