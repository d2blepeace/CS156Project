import wx

app = wx.App()

class UIFrame(wx.Frame):
    def __init__(self, parent, title):
        super().__init__(parent, title=title)
        self.panel = wx.Panel(self)
        self.label = wx.StaticText(self.panel, label="This is the beginning of the end.")


mainFrame = UIFrame(None, "Motion Controller")
mainFrame.Show()
app.MainLoop()