
# coding: utf-8

# In[29]:

import sys

from lr_question_answering import question_LR
from lr_topic_modelling import topic_LR


# In[36]:


def main():
    main_task = sys.argv[1]
    topic_file = sys.argv[2]

    if main_task == "topic":
        topic_LR(topic_file)

    elif main_task == "qa":
        question_LR(topic_file)

    else:
        print("sorry please try again with the commands qa and topic")

    sys.exit()


main()

# In[38]:


main()

