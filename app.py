from flask import Flask, render_template, request
import sqlite3
from datetime import datetime
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html', selected_from_date='', selected_to_date='', no_data=False)
@app.route('/attendance', methods=['POST'])
def attendance():
    selected_fdate = request.form.get('selected_from_date')
    selected_tdate = request.form.get('selected_to_date')
    if not selected_fdate or not selected_tdate:
        return render_template('index.html', selected_from_date=selected_fdate or '', selected_to_date=selected_tdate or '', no_data=True)
    try:
        selected_fdate_obj = datetime.strptime(selected_fdate, '%Y-%m-%d')
        selected_tdate_obj = datetime.strptime(selected_tdate, '%Y-%m-%d')
    except (ValueError, TypeError):
        return render_template('index.html', selected_from_date=selected_fdate or '', selected_to_date=selected_tdate or '', no_data=True)
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name, date, time FROM attendance WHERE date BETWEEN ? AND ?", (selected_fdate_obj.strftime('%Y-%m-%d'), selected_tdate_obj.strftime('%Y-%m-%d')))
    attendance_data = cursor.fetchall()
    conn.close()
    if not attendance_data:
        return render_template('index.html', selected_from_date=selected_fdate, selected_to_date=selected_tdate, no_data=True)
    return render_template('index.html', selected_from_date=selected_fdate, selected_to_date=selected_tdate, attendance_data=attendance_data)
if __name__ == '__main__':
    app.run(debug=True)
