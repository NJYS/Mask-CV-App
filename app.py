# -*- coding: utf-8 -*-

from flask import Flask, request, send_file  # 서버 구현을 위한 Flask 객체 import
from flask_restx import Api, Resource  # Api 구현을 위한 Api 객체 import
from run_model import run_model
from flask_cors import CORS

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
        print(request.json)
        lists[idx] = request.json.get('data')
        result = run_model(lists[idx])
        
        return {
            'todo_id': idx,
            'data': result
        }

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


#if __name__ == "__main__":
#    app.run(host='0.0.0.0', port='8000')
