' 生成带图标的工单智脑桌面快捷方式
' 用法：双击运行本 VBS，桌面会出现"工单智脑.lnk"
Set ws = CreateObject("WScript.Shell")
desktop = ws.SpecialFolders("Desktop")
Set s = ws.CreateShortcut(desktop & "\工单智脑.lnk")
s.TargetPath = "E:\WorkBuddy\ticketmind\start.bat"
s.WorkingDirectory = "E:\WorkBuddy\ticketmind"
s.IconLocation = "E:\WorkBuddy\ticketmind\assets\icon.ico,0"
s.Description = "工单智脑 TicketMind - 企业工单智能处理 Agent"
s.WindowStyle = 7  ' 最小化窗口
s.Save()
MsgBox "桌面快捷方式已创建：" & desktop & "\工单智脑.lnk", 64, "TicketMind"
