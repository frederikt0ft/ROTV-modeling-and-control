char buffer[10];

void setup() {
    Serial.begin(115200);
}

void loop() {
    if (Serial.available() > 0) {
        int num_bytes = Serial.readBytesUntil('\n', buffer, 3);
        buffer[num_bytes] = '\0';
        int num = atoi(buffer);
        // Do something with the integer num
        Serial.println(num);
        analogWrite(A0, num);
    }
}
