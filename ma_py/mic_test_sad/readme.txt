C������ ������ ��� VAD. ��� ���������� ������� ��� ����� SDR
� �������� ������� ���������� F1 ����.

� ����� 
.\data\_vad_test - ��� ����� �������� ������
.\data\_vad_test\_speech+prodigy_-10dB\mix - wav ����� � �������
.\data\_vad_test\_speech+prodigy_-10dB\config.cfg - ������ ����������� ������
.\data\_vad_test\_speech+prodigy_-10dB\cut_speech.seg  - ���� ����������� (���������� ��� ������ wave assistant) ��������� ������� ����.


\mic_test_vad - ��� ����� ������� �������.  ������ ��� ������������ 3 VAD 
1) ������� ��� �������� ��� ����
2) ������� ��� �������� ���  �� ����
3) ������� vad
---------------------------------------------------------------------------------------------------------------------------------------------

��� ������ ���������� ��������� ����� 
pip install pyinterval
---------------------------------------------------------------------------------------------------------------------------------------------


��� ����� ��������� VAD ����� ���������� �� BaseVAD (base_vad.py)


� ���������� ��������  ������ ������ �������� � ������� 
np.array([[time_begin, time_end], .... [time_begin, time_end]])

time_begin - ����� ������ ����
time_end  - ����� ��������� ����