# Middle-English-Language-Model
Middle English was the form of English spoken and written after the conquest of the Normans until the late fifteenth century. The Canterbury Tales is a book written during this period by Geoffrey Chaucer. This project is a demonstration of a language model in Middle English based on this book.

`web_scrape.py` stores the raw text into `text.txt`.
`train.py` cleans and preprocesses the text, makes a neural network model entirely in numpy ad trains it on the text.

`test.py` builds a sequence of following words based on the cosine distance with the input word.
