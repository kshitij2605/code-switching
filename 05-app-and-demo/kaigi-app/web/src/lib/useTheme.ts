import { useCallback, useEffect, useState } from 'react'

type Theme = 'dark' | 'light'
const KEY = 'kaigi-theme'

/** Dark-first theme toggle, persisted to localStorage and reflected on <html>. */
export function useTheme() {
  const [theme, setTheme] = useState<Theme>(() => {
    const saved = typeof localStorage !== 'undefined' ? localStorage.getItem(KEY) : null
    return saved === 'light' ? 'light' : 'dark'
  })

  useEffect(() => {
    const root = document.documentElement
    root.classList.toggle('dark', theme === 'dark')
    localStorage.setItem(KEY, theme)
  }, [theme])

  const toggle = useCallback(() => setTheme((t) => (t === 'dark' ? 'light' : 'dark')), [])

  return { theme, toggle }
}
