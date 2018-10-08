# coding: utf-8

# # Naive Bayes Sentiment Classification Program

# In[1]:

# unit test classifier
import sys
import time

import matplotlib.pyplot as plt
import util

amazon = "amazon_cells_labelled.txt"
imdb = "imdb_labelled.txt"
yelp = "yelp_labelled.txt"
test_file = sys.argv[1]  # input("enter test file: ")
model = None

if util.check_file():
    print("Loading Existing model....")
    model = util.load()
    print("Completed!")
else:
    print("Creating new model.....")

    if not util.check_file(amazon, imdb, yelp):
        print("Training files not available.")

    else:
        v2 = util.NB_DataHandler(amazon, imdb, yelp, quiet=False)
        res = v2.test()
        print("Initial Train Accuracy:\t{}%".format(str(round(res, 3))))

        # In[ ]:

        # test accuracy over several runs

        runs = 100
        run = None
        max_res = 0
        vals = []
        print("Selecting best model for unit testing out of {} runs...\n".format(runs))
        start = time.time()
        for i in range(1, runs + 1):
            if i % 10 == 0:
                t = round(time.time() - start, 3)
                print("run {}: {}s".format(i, t))
            v2 = util.NB_DataHandler(amazon, imdb, yelp)
            res = v2.test()
            if res >= max_res:
                max_res = res
                model = v2
                run = i
            vals.append(res)

        avg = sum(vals) / len(vals)
        finish = time.time() - start
        print("Max accuracy on run {}:\t{}% \nAverage accuracy: {}% \ntime: {}s"
              .format(run, round(max(vals), 3), round(avg, 3), round(finish, 3)))

        model.save()

        r = input("Do you wish to display Accuracy plot?\n[Y/N]: ")
        if r.lower() == "y":
            plt.plot(vals)
            plt.show()

        # In[ ]:
if model:
    print("beginning unit test on file: {}\n====================================================\n.......\n".format(
        test_file))

    test_res = model.unit_test(test_file)

    print("Unit test Complete!\nPlease open results_file.txt for results.\n")

    print("Do you wish to see a detailed report?\nThis will show how the model behaved on each sentences")
    i = input("[Y/N]: ")

    if i.lower() == "y":
        model.unit_test(test_file, True)
else:
    print("\nNo Loaded model.\nPlease ensure either training files or a valid model is available and try again.")

print("End of program")
