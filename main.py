from flask import Flask,request,jsonify
from rag_with_palm import RAGPaLMQuery
from flask_cors import CORS

rag_palm_query_instance = RAGPaLMQuery()

app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat():
    if request.method == "POST":
        data = request.get_json(force=True)  # Added 'force=True'
        k = data.get("message")
        respons = rag_palm_query_instance.query_response(k)
        res = {
            "msg": str(respons)
        }
        return jsonify(res), 200



if __name__ == "__main__":
    app.run(debug=True)