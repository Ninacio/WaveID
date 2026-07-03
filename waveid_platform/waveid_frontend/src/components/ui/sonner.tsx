import { Toaster as Sonner, type ToasterProps } from "sonner"

import { useTheme } from "@/components/theme/theme-provider"

function Toaster({ ...props }: ToasterProps) {
  const { theme } = useTheme()
  return (
    <Sonner
      theme={theme}
      className="toaster group"
      toastOptions={{
        style: {
          background: "var(--card)",
          color: "var(--card-foreground)",
          border: "1px solid var(--border)",
        },
      }}
      {...props}
    />
  )
}

export { Toaster }
