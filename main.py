from flask import Flask,request,jsonify
from rag_with_palm import RAGPaLMQuery

rag_palm_query_instance = RAGPaLMQuery()

app = Flask(__name__)

@app.route("/chat",methods=["POST"])
def chat():
    if request.method == "POST":
        data = request.get_json()
        k = data.get("respon")
        respons = rag_palm_query_instance.query_response(k)
        res = {
            "msg":str(respons)
        }
        #print(res.get("msg"))
        return jsonify(res),200


if __name__ == "__main__":
    app.run(debug=True)