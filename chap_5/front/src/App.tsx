import { Routes, Route } from 'react-router'
import {WelcomeGridScreen} from "./features/welcome_grid/screens/WelcomeGridScreen.tsx";
import {MovieDetailScreen} from "./features/detail/screens/MovieDetailScreen.tsx";

function App() {
    return (
        <Routes>
            <Route path="/" element={<WelcomeGridScreen />} />
            <Route path="/movie/:id" element={<MovieDetailScreen />} />
        </Routes>
    )
}

export default App