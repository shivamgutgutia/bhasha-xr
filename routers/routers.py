from flask import Blueprint
from controllers import *
router = Blueprint('router', __name__)

router.add_url_rule('/', 'home', home,methods=["GET"])
router.add_url_rule('/ping', 'ping', ping,methods=["GET"])
router.add_url_rule("/api/evaluate","evaluate",evaluate,methods=["POST"])