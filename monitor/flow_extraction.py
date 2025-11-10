# flask -> sqlalchemy 
# monitoring data, when connection terminates we send it to the server's database and we use it later

# prerequisites: windows Npcap for windows packet capture
# know your network interface name (probably "Wi-Fi" if on WiFi or "Eth1" if on ethernet or so on should check network configs)

# CLI Command:
# cicflowmeter -i "Wi-Fi" -u http://127.0.0.1:8000/write_flow