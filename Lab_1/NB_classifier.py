import util

amazon = "amazon_cells_labelled.txt"
imdb = "imdb_labelled.txt"
yelp = "yelp_labelled.txt"

v2 = util.Data(amazon, imdb, yelp, quiet=False)
res = v2.test()
print(res)
