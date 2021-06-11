from flask import Flask, request  # 서버 구현을 위한 Flask 객체 import
from flask_restx import Api, Resource  # Api 구현을 위한 Api 객체 import

app = Flask(__name__)  # Flask 객체 선언, 파라미터로 어플리케이션 패키지의 이름을 넣어줌.
api = Api(app)  # Flask 객체에 Api 객체 등록


lists = {}
count = 1


@api.route('/hello')  # 데코레이터 이용, '/hello' 경로에 클래스 등록
class HelloWorld(Resource):
    def get(self):  # GET 요청시 리턴 값에 해당 하는 dict를 JSON 형태로 반환
        return {"hello": "world!"}

@api.route('/masks')
class TodoPost(Resource):
    def post(self):
        global count
        global lists
        
        idx = count
        count += 1
        lists[idx] = request.json.get('data')
        
        return {
            'todo_id': idx,
            'data': lists[idx]
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

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)