# -*- coding: utf-8 -*-

from flask import Flask, request, send_file  # 서버 구현을 위한 Flask 객체 import
from flask_restx import Api, Resource  # Api 구현을 위한 Api 객체 import
from run_model import inference_image
from flask_cors import CORS
import json

app = Flask(__name__)  # Flask 객체 선언, 파라미터로 어플리케이션 패키지의 이름을 넣어줌.
api = Api(app)  # Flask 객체에 Api 객체 등록
CORS(app)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

lists = {}
count = 1

@api.route('/hello')  # 데코레이터 이용, '/hello' 경로에 클래스 등록
class HelloWorld(Resource):
    def get(self):  # GET 요청시 리턴 값에 해당 하는 dict를 JSON 형태로 반환
        return {"hello111": "world!"}

@api.route('/masks')
class TodoPost(Resource):
    def post(self):
        global count
        global lists
        
        idx = count
        count += 1
        lists[idx] = request.json.get('data')
        # lists[idx] = "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"
        result = inference_image(lists[idx])
        
        return json.dumps(result)

@api.route('/masks/<int:todo_id>')
class TodoSimple(Resource):
    def get(self, todo_id):
        return {
            'todo_id': todo_id,
            'data': lists[todo_id]
        }

    def put(self, todo_id):
        lists[todo_id] = request.json.get('data')
        return {
            'todo_id': todo_id,
            'data': lists[todo_id]
        }
    
    def delete(self, todo_id):
        del lists[todo_id]
        return {
            "delete" : "success"
        }

# tensorflow js
@api.route('/download-test')
class Download_file(Resource):
    def get(self):
        #return {"hello":"world"}
        return send_file("./converted/model.json", as_attachment=True, cache_timeout=0)
        #return send_file("~/mask-cv-back/test.txt", as_attachment=True)

@api.route('/group1-shard1of21.bin')
class bin_file_1(Resource):
    def get(self):
        return send_file("./converted/group1-shard1of21.bin", as_attachment=True, cache_timeout=0)

@api.route('/group1-shard2of21.bin')
class bin_file_1(Resource):
    def get(self):
        return send_file("./converted/group1-shard2of21.bin", as_attachment=True, cache_timeout=0)

@api.route('/group1-shard3of21.bin')
class bin_file_1(Resource):
    def get(self):
        return send_file("./converted/group1-shard3of21.bin", as_attachment=True, cache_timeout=0)

@api.route('/group1-shard4of21.bin')
class bin_file_1(Resource):
    def get(self):
        return send_file("./converted/group1-shard4of21.bin", as_attachment=True, cache_timeout=0)

@api.route('/group1-shard5of21.bin')
class bin_file_1(Resource):
    def get(self):
        return send_file("./converted/group1-shard5of21.bin", as_attachment=True, cache_timeout=0)

@api.route('/group1-shard6of21.bin')
class bin_file_1(Resource):
    def get(self):
        return send_file("./converted/group1-shard6of21.bin", as_attachment=True, cache_timeout=0)

@api.route('/group1-shard7of21.bin')
class bin_file_1(Resource):
    def get(self):
        return send_file("./converted/group1-shard7of21.bin", as_attachment=True, cache_timeout=0)

@api.route('/group1-shard8of21.bin')
class bin_file_1(Resource):
    def get(self):
        return send_file("./converted/group1-shard8of21.bin", as_attachment=True, cache_timeout=0)

@api.route('/group1-shard9of21.bin')
class bin_file_1(Resource):
    def get(self):
        return send_file("./converted/group1-shard9of21.bin", as_attachment=True, cache_timeout=0)

@api.route('/group1-shard10of21.bin')
class bin_file_1(Resource):
    def get(self):
        return send_file("./converted/group1-shard10of21.bin", as_attachment=True, cache_timeout=0)

@api.route('/group1-shard11of21.bin')
class bin_file_1(Resource):
    def get(self):
        return send_file("./converted/group1-shard11of21.bin", as_attachment=True, cache_timeout=0)

@api.route('/group1-shard12of21.bin')
class bin_file_1(Resource):
    def get(self):
        return send_file("./converted/group1-shard12of21.bin", as_attachment=True, cache_timeout=0)

@api.route('/group1-shard13of21.bin')
class bin_file_1(Resource):
    def get(self):
        return send_file("./converted/group1-shard13of21.bin", as_attachment=True, cache_timeout=0)

@api.route('/group1-shard14of21.bin')
class bin_file_1(Resource):
    def get(self):
        return send_file("./converted/group1-shard14of21.bin", as_attachment=True, cache_timeout=0)

@api.route('/group1-shard15of21.bin')
class bin_file_1(Resource):
    def get(self):
        return send_file("./converted/group1-shard15of21.bin", as_attachment=True, cache_timeout=0)

@api.route('/group1-shard16of21.bin')
class bin_file_1(Resource):
    def get(self):
        return send_file("./converted/group1-shard16of21.bin", as_attachment=True, cache_timeout=0)

@api.route('/group1-shard17of21.bin')
class bin_file_1(Resource):
    def get(self):
        return send_file("./converted/group1-shard17of21.bin", as_attachment=True, cache_timeout=0)

@api.route('/group1-shard18of21.bin')
class bin_file_1(Resource):
    def get(self):
        return send_file("./converted/group1-shard18of21.bin", as_attachment=True, cache_timeout=0)

@api.route('/group1-shard19of21.bin')
class bin_file_1(Resource):
    def get(self):
        return send_file("./converted/group1-shard19of21.bin", as_attachment=True, cache_timeout=0)

@api.route('/group1-shard20of21.bin')
class bin_file_1(Resource):
    def get(self):
        return send_file("./converted/group1-shard20of21.bin", as_attachment=True, cache_timeout=0)

@api.route('/group1-shard21of21.bin')
class bin_file_1(Resource):
    def get(self):
        return send_file("./converted/group1-shard21of21.bin", as_attachment=True, cache_timeout=0)


if __name__ == "__main__":
   app.run(host='0.0.0.0', port='8000')
