/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#0b0d10",
          900: "#11141a",
          800: "#171b22",
          700: "#1f242d",
          600: "#2a313c",
          500: "#3a424f",
          400: "#5a6271",
          300: "#8a93a3",
          200: "#c2c8d3",
          100: "#e6e9ef",
        },
        accent: {
          500: "#5b8cff",
          400: "#7aa3ff",
          300: "#a5c0ff",
        },
      },
      fontFamily: {
        mono: ["SF Mono", "JetBrains Mono", "Menlo", "monospace"],
      },
    },
  },
  plugins: [],
};
