import os
import validation

def main():
    files = os.listdir('./data')

    for f in files:
        validation.main(['FILLER', f'./data/{f}', '10', '.25', '0'])

if __name__ == '__main__':
    main()