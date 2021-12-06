import getopt, sys


def user_arg():
    load_models, epochs, gamma, len_trajectory, batch_size, lr = False, int(10e7), 0.99, 10, 4, [30e-5,1e-4,1e-4]
    argument_list = sys.argv[1:]

    # load, epochs: int, gamma: float, trajectory: int, batches:int, 
    options = "le:g:t:b:"
    
    # learning rates: List[float] 
    long_options = ["LR ="]
    
    try:
        # Parsing argument
        arguments, _values = getopt.getopt(argument_list, options, long_options)

        # checking each argument
        for currentArgument, currentValue in arguments:
        
            if currentArgument in ("-l"):
                print("Will load models") 
                load_models = True

            elif currentArgument in ("-e"):
                epochs = int(currentValue)
                print(f"Training for {epochs} epochs")

            elif currentArgument in ("-g"):
                gamma = float(currentValue)
                print(f"Discount factor γ set to {gamma}")

            elif currentArgument in ("-t"):
                len_trajectory = int(currentValue)
                print(f"Trajectory length {len_trajectory}")

            elif currentArgument in ("-b"):
                batch_size = int(currentValue)
                print(f"Using batchsize {batch_size}")


            elif currentArgument.strip() in ("--LR"):
                lr = []
                for val in currentValue.split(','):
                    lr.append(float(val))
                print (("Using learning rates (% s)") % (lr))

    except getopt.error as err:
        # output error, and return with an error code
        print (str(err))

    return load_models, epochs, gamma, len_trajectory, batch_size, lr 

if __name__ == "__main__":
    print(user_arg())
