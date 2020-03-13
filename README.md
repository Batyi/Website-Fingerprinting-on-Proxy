# Website_Fingerprinting_Proxy
Website fingerprinting (WFP) is a special type of traffic analysis attack where an adversary attempts to identify which page a user is visiting by analyzing patterns of encrypted communication. The attacker is located between the user and the proxy server. The adversary is not able to decrypt the observed traffic. Thus, the adversary can only see the meta-data (e.g., packet size, direction) of the communication. WFP
relies on the classification of TCP/IP traces to a given set of websites. The key idea is to train a
machine learning model with traces from a known website.

License
----
MIT