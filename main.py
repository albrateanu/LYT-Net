import argparse
import subprocess

def run_train(dataset):
    subprocess.run(['python', './scripts/train.py', '--dataset', dataset])

def run_test(dataset):
    subprocess.run(['python', './scripts/test.py', '--dataset', dataset])

def run_complexity_check():
    subprocess.run(['python', './scripts/complexity_check.py'])

def main():
    parser = argparse.ArgumentParser(description='Run project scripts')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true', help='Run training script')
    group.add_argument('--test', action='store_true', help='Run testing script')
    group.add_argument('--complexity', action='store_true', help='Run complexity check script')
    
    parser.add_argument('--dataset', type=str, help='Dataset name for training')

    args = parser.parse_args()

    if args.train:
        if args.dataset:
            run_train(args.dataset)
        else:
            print("Please specify a dataset for training using '--dataset [LOLv1/LOLv2_Real/LOLv2_Synthetic]'")
    elif args.test:
        if args.dataset:
            run_test(args.dataset)
        else:
            print("Please specify a dataset for testing using '--dataset [LOLv1/LOLv2_Real/LOLv2_Synthetic]'")
    elif args.complexity:
        run_complexity_check()

if __name__ == '__main__':
    main()
