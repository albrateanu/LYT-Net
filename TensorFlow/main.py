import argparse
import subprocess

def run_train(dataset):
    subprocess.run(['python', './scripts/train.py', '--dataset', dataset])

def run_test(dataset, weights, gtmean=False):
    cmd = ['python', './scripts/test.py', '--dataset', dataset, '--weights', weights]
    if gtmean:
        cmd.append('--gtmean')
    subprocess.run(cmd)


def run_complexity_check(shape='(256,256,3)'):
    subprocess.run(['python', './scripts/complexity_check.py', '--shape', shape])

def main():
    parser = argparse.ArgumentParser(description='Run project scripts')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true', help='Run training script')
    group.add_argument('--test', action='store_true', help='Run testing script')
    group.add_argument('--complexity', action='store_true', help='Run complexity check script')
    
    parser.add_argument('--gtmean', action='store_true', help='Use GT Mean for evaluation.')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--weights', type=str, help='Dataset name')
    parser.add_argument('--shape', type=str, help='Input shape')

    args = parser.parse_args()

    if args.train:
        if args.dataset:
            run_train(args.dataset)
        else:
            print("Please specify a dataset for training using '--dataset [LOLv1/LOLv2_Real/LOLv2_Synthetic]'")
    elif args.test:
        if args.dataset and args.weights:
            run_test(args.dataset, args.weights, args.gtmean)
        if not args.dataset:
            print("Please specify a dataset for testing using '--dataset [LOLv1/LOLv2_Real/LOLv2_Synthetic]'")
        if not args.weights:
            print("Please specify a path to a '.h5' file containing model weights.")
    elif args.complexity:
        if args.shape:
            shape = args.shape[1:len(args.shape)-1]
            shape = shape.split(',')
            shape = [int(x) for x in shape]
            if len(shape) != 3:
                print("Please provide an input shape as a tuple of the form '(H,W,C)'")
                return
            run_complexity_check(args.shape)
        else:
            run_complexity_check()

if __name__ == '__main__':
    main()
