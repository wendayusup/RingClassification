"""
QUICK LAUNCHER: STAGE 6 WEB INSPECTION DASHBOARD
------------------------------------------------
Launches the Flask web server for industrial bearing defect inspection.
"""

from step06_web_app.app import app

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("   BEARING DEFECT INSPECTION WEB APPLICATION RUNNING")
    print("=" * 60)
    print("   URL: http://127.0.0.1:5000")
    print("   Press CTRL+C in terminal to stop server.")
    print("=" * 60 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
