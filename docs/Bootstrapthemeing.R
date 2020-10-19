library(sass)
library(bootstraplib)

bs_theme_new(bootswatch = "flatly")
bs_theme_accent_colors(primary = "#3E4D54", secondary = "#3fbfba", 
                       success = "#3fbfba", info = "#008ba6", warning = "#f29c20", 
                       danger = "#cedb00")
bs_theme_add_variables(black = "#58585B")
bs_theme_fonts(base = "'ciutadella', light", heading = "'nexa', regular")

bs_theme_preview()



bs_theme_get()

as_sass(bs_theme_get())

sass(
  input = bootstrap_sass(bs_theme_get()),
  output = "RPGTheme.css",
  write_attachments = FALSE
)

bootstrap_sass(bs_theme_get())

output:
  html_document:
  theme: flatly