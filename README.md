🧠 Epileptic Seizure Detection & Alert System

A real-time Epileptic Seizure Detection System that analyzes EEG signals using Machine Learning (Random Forest) and Signal Processing (Wavelet Transform). When a seizure is detected, the system instantly triggers multi-channel alerts via Telegram and Email to caregivers and doctors.

🔗 Live Demo: Click Here to Launch App

🚀 Key Features

Real-Time Prediction: Classifies EEG signals as "Seizure" or "Normal" in milliseconds.

Advanced Signal Processing: Uses Discrete Wavelet Transform (DWT) and Hurst Exponent for feature extraction.

Multi-Channel Alerts:

📱 Telegram Bot: Sends instant push notifications to a family/doctor group chat.

📧 Email: Sends detailed reports to emergency contacts.

🔊 Sound Alarm: Plays a loud alert sound locally for immediate attention.

Interactive Dashboard: Built with Streamlit for easy data visualization and manual testing.

📸 Project Screenshots

1. Main Dashboard & Normal Activity

The user interface allows uploading CSV files or manual input. When normal brain activity is detected, the system shows a green status.

2. Seizure Detected!

When a seizure pattern is identified, the system flashes a red warning and triggers the alert protocols.

3. Telegram Alert (Mobile View)

Instant notification received on a mobile phone via the Telegram Bot, showing the exact time and row index of the seizure.

4. Email Alert

An automated email is sent to the registered caregiver with urgent details.

5. Manual Input Testing

Allows doctors/researchers to manually paste EEG values to test specific signal patterns.

🛠️ Tech Stack

Frontend: Streamlit

Machine Learning: Scikit-Learn (Random Forest Classifier)

Signal Processing: PyWavelets (Wavelet Transform), Hurst (Hurst Exponent)

Alerts: Python requests (Telegram API), smtplib (Gmail SMTP)

Deployment: Streamlit Community Cloud

⚙️ Installation & Setup

1. Clone the Repository

git clone [https://github.com/your-username/epilepsy-detection-system.git](https://github.com/your-username/epilepsy-detection-system.git)
cd epilepsy-detection-system


2. Install Dependencies

pip install -r requirements.txt


3. Configure Secrets (.env)

Create a .env file in the root directory and add your keys:

TELEGRAM_BOT_TOKEN="your_telegram_bot_token"
TELEGRAM_CHAT_ID="-100xxxxxxxxxx"  # Your Group ID
EMAIL_SENDER="your_email@gmail.com"
EMAIL_PASSWORD="your_16_digit_app_password"
EMAIL_RECEIVER="doctor_email@gmail.com"


4. Run the App

streamlit run app.py


📊 Dataset Details

This project uses the UCI Epileptic Seizure Recognition Data Set.

Total Rows: 11,500

Columns: 179 (178 EEG features + 1 Target label)

Sampling Rate: 178 Hz (1 second of recording per row)

Classes:

1: Seizure Activity (Epileptic)

0: Normal / Tumor Area / Eyes Closed (Non-Seizure)

🔮 Future Scope

Integrate IoT Headsets (like Emotiv/OpenBCI) for live brainwave streaming.

Add GPS Location to the alert message for emergency ambulances.

Implement Deep Learning (LSTM/CNN) for potentially higher accuracy.

👨‍💻 Author

Rohan & Team
College Project - 2025

Disclaimer: This tool is a prototype for educational purposes and should not replace professional medical diagnosis.


### Instructions to Add Screenshots
1.  Create a folder named `screenshots` inside your project folder.
2.  Rename your screenshot files to match the names I used in the README:
    * `Screenshot (644).png` -> `normal_activity.png`
    * `Screenshot (645).png` -> `seizure_detected.png`
    * `Screenshot (646).png` -> `telegram_alert.jpg`
    * `Screenshot (647).png` -> `email_alert.jpg`
    * `Screenshot (648).png` -> `manual_input.png`
3.  Move these renamed images into the `screenshots` folder.
4.  Upload everything to GitHub.

This README is professional, complete, and ready for submission!
