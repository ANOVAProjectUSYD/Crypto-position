from flask import Flask
from flask import render_template
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

from config import Config 


app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)
migrate = Migrate(app, db)

from models import *

@app.route("/")
def index():
    return render_template("index.html")

@app.shell_context_processor
def make_shell_context():
    """Includes objects we want to use by default when we run flask shell"""
    return {'db': db, 'Bitcoin': Bitcoin}

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000,debug=True)
