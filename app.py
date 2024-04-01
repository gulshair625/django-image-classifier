import json

from flask import Flask
from flask_restful import Api, Resource, request

from handler import predict

app = Flask(__name__)
api = Api(app)


class ImageClassifier(Resource):

    def post(self):
        return predict(json.loads(request.data), [])


api.add_resource(ImageClassifier, '/predict')

if __name__ == "__main__":
    app.run(debug=True)
