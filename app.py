from flask import Flask, render_template, request

app = Flask(__name__, static_url_path="", static_folder="templates", template_folder="templates")

from lsh import performLSHcorpus, performLSHquery, find_similar_docs, getDataForDocumnetById


# corpusBucket = None
# dictShinglesId = None
corpusBucket, dictShinglesId = performLSHcorpus()
def get_result(query_inp):
    global corpusBucket
    global dictShinglesId

    # if corpusBucket is None or dictShinglesId is None:
    #     corpusBucket, dictShinglesId = performLSHcorpus()
    queryBucket = performLSHquery(query_inp, dictShinglesId)
    docIdList = find_similar_docs(queryBucket, corpusBucket)
    # for item in docIdList:
    #     print(getDataForDocumnetById(item))

    list_of_docs = []
    for item in docIdList:
        list_of_docs.append(getDataForDocumnetById(item)[0])


    return list_of_docs


# disable favicon.ico
@app.route("/favicon.ico")
def favicon():
    return ""


@app.route("/", methods=["GET", "POST"])
def hello_world():
    # make a variable to store the input text
    if request.method == "POST":
        # store the input text in the variable
        input_text = request.form["query"]
        return render_template("index.html", result_op=get_result(input_text))
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
