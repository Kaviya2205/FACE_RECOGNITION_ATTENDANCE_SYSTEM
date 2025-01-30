import mysql.connector
import datetime

conn = mysql.connector.connect(host="localhost", user="root", password="password", database="attendance_db")
cursor = conn.cursor()

def mark_attendance(name):
    time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO attendance (name, timestamp) VALUES (%s, %s)", (name, time_now))
    conn.commit()
