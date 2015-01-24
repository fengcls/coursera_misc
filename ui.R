library(shiny)
shinyUI(pageWithSidebar(
  # Application title
  headerPanel("Bonus Point used in Torrent tracker"),
  sidebarPanel(
    h5('This app is used to calculate the bonus point for supporting a torrent for an hour,
       what it need as input is in the sidebar, when ready, press submit, the value you type in
       will show in the right, and so will the bonus calculation'),
    numericInput('T0', 'How long the torrent lasts (week)', 10, min = 0, max = 100, step = 4),
    numericInput('S', 'The size of the file (GB)', 10, min = 0, max = 20, step = 1),
    numericInput('N', 'The amount of users', 10, min = 0, max = 100, step = 5),
    submitButton('Submit')
  ),
  mainPanel(
    h3('Results of calculation'),
    h4('You entered a torrent last'),
    verbatimTextOutput("T0"),
    h4('weeks, with a size of'),
    verbatimTextOutput("S"),
    h4('GB, torrented by'),
    verbatimTextOutput("N"),
    h4('users'),
    h4('Which resulted in a bonus every hour: '),
    verbatimTextOutput("bonus")
  ) )
)