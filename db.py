#!/usr/bin/python
from configparser import ConfigParser
import psycopg2
import requests
import json

conn = None


def config(filename='database.ini', section='postgresql'):
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)

    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception(
            'Section {0} not found in the {1} file'.format(section, filename))

    return db


def connect():
    global conn
    """ Connect to the PostgreSQL database server """
    # conn = None
    try:
        # read connection parameters
        params = config()

        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)
        # create a cursor
        # cur = conn.cursor()

        # execute a statement
        # print('PostgreSQL database version:')
        # cur.execute('SELECT version()')

        # display the PostgreSQL database server version
        # db_version = cur.fetchone()
        # print(db_version)

       # close the communication with the PostgreSQL
        # cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            pass
            # conn.close()
            # print('Database connection closed.')


def raise_complain(against_user_id, details, created_by, parking_lot_id=None, parking_space_id=None):
    global conn
    """ insert a new vendor into the vendors table """
    sql = """INSERT INTO public.parking_complain(
	against_user_id, parking_lot_id, parking_space_id, details, created_by)
	VALUES ( %s, %s, %s, %s, %s)
    returning id"""
    # conn = None
    if conn is None:
        # read database configuration
        params = config()
        # connect to the PostgreSQL database
        conn = psycopg2.connect(**params)

    insert_id = None
    try:
        # create a new cursor
        cur = conn.cursor()
        # execute the INSERT statement
        cur.execute(sql, (against_user_id, parking_lot_id,
                          parking_space_id, details, created_by))
        # get the generated id back
        insert_id = cur.fetchone()[0]
        # commit the changes to the database
        conn.commit()
        # close communication with the database
        cur.close()

        send_push_noti(against_user_id)

    except (Exception, psycopg2.DatabaseError) as error:
        print("error", error)
        return error
    finally:
        pass
        # if conn is not None:
        #     conn.close()

    return insert_id


def send_push_noti(user_id):
    global conn
    sql = """SELECT noti_token FROM public."user" where id=%s"""
    if conn is None:
        conn = connect()

    insert_id = None
    try:
        # create a new cursor
        cur = conn.cursor()
        # execute the INSERT statement
        cur.execute(sql, (user_id,))
        # get the generated id back
        rows = cur.fetchall()
        print("The number of parts: ", cur.rowcount)

        token = None
        for row in rows:
            print(row[0])
            token = row[0]
        if token is not None:
            request_fire_base(token)
        cur.close()
        # commit the changes to the database
        conn.commit()
        # close communication with the database
    # except (Exception, psycopg2.DatabaseError) as error:
    except (psycopg2.DatabaseError) as error:
        print("error", error)
        return error
    finally:
        pass
        # if conn is not None:
        #     conn.close()

    return insert_id


def save_register_noti(user_id, token):
    global conn
    sql = """UPDATE public."user"
	SET noti_token=%s
	WHERE firebase_id=%s
    returning id"""
    if conn is None:
        connect()

    insert_id = None
    try:
        # create a new cursor
        cur = conn.cursor()
        # execute the INSERT statement
        cur.execute(sql, (token, user_id))
        # get the generated id back
        insert_id = cur.fetchone()[0]
        # commit the changes to the database
        conn.commit()
        # close communication with the database
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print("error", error)
        return error
    finally:
        pass
        # if conn is not None:
        #     conn.close()

    return insert_id

def request_fire_base(token):
    url = "https://fcm.googleapis.com/fcm/send"
    payload = json.dumps({
        "notification": {
            "title": "Parkign Issue",
            "body": "It's seems your car is double parked, kindly park properly",
            "click_action": "http://localhost:4200/defaultLayout/scanCar"
        },
        "to": token
    })
    headers = {
        'Authorization': "key=AAAAWe_xG2k:APA91bGK8imO-seV6g8vB6kiUNqsffRs7M-3u68k_ViCp8CLL5Ko0tKCOfHuZUFTBOMkaaW4opX8G1w6j1STvP1G1KqEMGy9J-3sr9RFO3ZQCEJRy65ycva2p7XmSb-rFT_iY3SAGPSR",
        'Content-Type': "application/json",
        'cache-control': "no-cache"
    }

    response = requests.request("POST", url, data=payload, headers=headers)
    print(response.text)
    if response.status_code != 200:
        return False
    return True
