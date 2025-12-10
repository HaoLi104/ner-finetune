import json
import os
import sys
import time
from datetime import datetime

from peewee import *
from playhouse.shortcuts import ReconnectMixin

db_conf = {
    "host": "127.0.0.1",
    "port": 3306,
    "database": "hx",
    "username": "wy",
    "password": "1qaz2wsX",
}


class ReconnectMySQLDatabase(ReconnectMixin, MySQLDatabase):
    pass


db_conf = {
    "host": db_conf["host"],
    "port": db_conf["port"],
    "database": db_conf["database"],
    "username": db_conf["username"],
    "password": db_conf["password"],
}

db_conn = ReconnectMySQLDatabase(
    db_conf["database"],
    host=db_conf["host"],
    port=db_conf["port"],
    user=db_conf["username"],
    passwd=db_conf["password"],
)


class JSONField(TextField):
    def db_value(self, value):
        return json.dumps(value) if value else None

    def python_value(self, value):
        if value is not None:
            return json.loads(value)


class BaseModel(Model):
    class Meta:
        database = db_conn


class hx_pic_result(BaseModel):
    user_id = CharField()
    pic_id = CharField()
    image_url = CharField()
    # ALTER TABLE `hx_pic_result` ADD `image_width` INT NOT NULL DEFAULT '0' AFTER `image_url`, ADD `image_height` INT NOT NULL DEFAULT '0' AFTER `image_width`;
    image_width = IntegerField(default=0)
    image_height = IntegerField(default=0)

    image_url_desensitive = CharField()
    scale_factor = FloatField(default=1)
    # ALTER TABLE `hx_pic_result` ADD `rotate_angle` DOUBLE NOT NULL DEFAULT '0' COMMENT '原图旋转角度' AFTER `scale_factor`;
    rotate_angle = FloatField(default=1)

    ocr_res = JSONField(null=True)
    # ALTER TABLE `hx_pic_result` ADD `category` VARCHAR(100) NULL DEFAULT NULL AFTER `ocr_res`;
    category = CharField(default=None, null=True)
    # ALTER TABLE `hx_pic_result` ADD `text_dup_id` BIGINT NULL DEFAULT NULL AFTER `ocr_res`;
    text_dup_id = BigIntegerField(default=None, null=True)
    # ALTER TABLE `hx_pic_result` ADD `disease_type` VARCHAR(100) NULL DEFAULT NULL AFTER `text_dup_id`, ADD `pic_annotation` JSON NULL DEFAULT NULL COMMENT '直接提取结果，未经规则' AFTER `disease_type`, ADD `pic_annotation_custom` JSON NULL DEFAULT NULL AFTER `pic_annotation`;
    disease_type = CharField(default=None, null=True)
    # ALTER TABLE `hx_pic_result` ADD `disease_time` VARCHAR(255) NULL DEFAULT NULL AFTER `disease_type`;
    disease_time = CharField(default=None, null=True)
    # 直接提取结果，未经规则
    pic_annotation = JSONField(default=None, null=True)
    pic_annotation_custom = JSONField(default=None, null=True)

    # should use JSONField -- historical problem
    pic_res = TextField(default=None, null=True)
    pic_res_custom = JSONField(default=None, null=True)
    cache_key = CharField()

    hash_code = CharField()
    token_in = IntegerField(default=0)
    token_out = IntegerField(default=0)
    total_token = IntegerField(default=0)
    token_cost = FloatField(default=0)
    token_in_cost = FloatField(default=0)
    token_out_cost = FloatField(default=0)
    channel = CharField()
    create_time = DateTimeField(default=datetime.now)


def model_to_dict(model):
    # 确保我们有一个字段对象列表
    fields = getattr(model._meta, "fields", [])
    # 使用字典推导式创建字典
    return {
        field.name: getattr(model, field.name)
        for field in fields
        if isinstance(field, object)
    }
