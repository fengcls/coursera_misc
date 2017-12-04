library(shiny)
bonustorr <- function(T0,S,N)
{
tmp_A <- (1-exp(-2.3*T0/4))*S*(1+sqrt(2)*exp(-2.3*(N-1)/6))
bonus <- 100*2/pi*atan(tmp_A/450)
return(bonus)
}
shinyServer( function(input, output) {
  output$T0<-renderPrint({input$T0})
  output$S<-renderPrint({input$S})
  output$N<-renderPrint({input$N})
  output$bonus<-renderPrint({bonustorr(input$T0,input$S,input$N)})
} )