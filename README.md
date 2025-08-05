# 本機測試Github Actions
- 使用套件：act
- 測試方式：
--1.啟動Docker
--2.到repo根目錄
--3.開啟wsl
--4.執行指令 sh ./scripts/env2secrets.sh
--5.執行指令 act workflow_dispatch -P ubuntu-latest=catthehacker/ubuntu:act-latest --secret-file .secrets